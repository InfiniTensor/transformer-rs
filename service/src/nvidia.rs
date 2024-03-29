﻿use crate::{session, Command};
use common::utok;
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::Path,
    sync::{mpsc::Receiver, Arc, Mutex},
    time::Instant,
};
use transformer_cpu::{Llama2, Memory, SampleArgs};
use transformer_nvidia::{
    cuda::{ContextResource, Device, Stream},
    LayerCache, Request, Transformer,
};

pub fn task(
    device: Device,
    model_dir: impl AsRef<Path>,
    sample: Arc<Mutex<SampleArgs>>,
    receiver: Receiver<Command>,
) {
    device.set_mempool_threshold(u64::MAX);
    let model_dir = model_dir.as_ref();

    let time = Instant::now();
    let config = File::open(model_dir.join("config.json")).unwrap();
    let mut safetensors = File::open(model_dir.join("model.safetensors")).unwrap();
    info!("open file {:?}", time.elapsed());

    device.context().apply(|ctx| {
        let time = Instant::now();
        let host = ctx.malloc_host::<u8>(safetensors.metadata().unwrap().len() as _);
        let mut host = host.sporulate();
        safetensors.read_exact(&mut host).unwrap();
        drop(safetensors);
        info!("read to host {:?}", time.elapsed());

        let compute = ctx.stream();
        let transfer = ctx.stream();

        let time = Instant::now();
        let host = Memory::load_safetensors(config, host, false).unwrap();
        let max_seq_len = host.max_position_embeddings();
        let eos = host.eos_token_id();
        let transformer = Transformer::new(Box::new(host), usize::MAX, &transfer);
        info!("build model host: {:?}", time.elapsed());

        let mut sessions = HashMap::new();

        while let Ok(cmd) = receiver.recv() {
            match cmd {
                Command::Infer {
                    id,
                    prompt,
                    responsing,
                } => {
                    let ctx = sessions
                        .entry(id)
                        .or_insert_with_key(|&id| SessionContext::new(&transformer, id, &transfer));

                    let t0 = Instant::now();
                    let mut token = transformer.decode(
                        vec![ctx.request(&prompt, max_seq_len)],
                        &sample.lock().unwrap(),
                        &compute,
                        &transfer,
                    )[0]
                    .1;
                    let t1 = Instant::now();
                    let mut len = 0;
                    while token != eos {
                        responsing.send(token).unwrap();
                        token = transformer.decode(
                            vec![ctx.request(&[token], max_seq_len)],
                            &sample.lock().unwrap(),
                            &compute,
                            &transfer,
                        )[0]
                        .1;
                        len += 1;
                    }
                    let t2 = Instant::now();
                    info!(
                        "First token delay: {:?}, average speed = {:?}/tok",
                        t1 - t0,
                        (t2 - t1).div_f32(len as _)
                    );
                }
                Command::Drop { id } => {
                    sessions.remove(&id);
                }
            }
        }
    });
}

struct SessionContext<'a>(session::SessionContext<LayerCache<'a>>);

impl<'a> SessionContext<'a> {
    #[inline]
    fn new(transformer: &Transformer, id: usize, stream: &'a Stream) -> Self {
        Self(session::SessionContext::new(
            transformer.new_cache(stream),
            id,
        ))
    }

    #[inline]
    fn request(&mut self, tokens: &[utok], max_seq_len: usize) -> Request<'_, 'a, usize> {
        let pos = self.0.request(tokens, max_seq_len);
        Request::new(
            self.0.id,
            &self.0.cache_map[pos..],
            &mut self.0.cache,
            pos as _,
            true,
        )
    }
}
