mod cpu;
#[cfg(detected_cuda)]
mod nvidia;
mod session;
mod template;

use common::utok;
use session::SessionComponent;
use std::{
    path::Path,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};
use template::Template;
use tokenizer::{BPECommonNormalizer, Normalizer, Tokenizer, VocabTxt, BPE};
use transformer_cpu::SampleArgs;

pub use session::Session;

#[macro_use]
extern crate log;

pub struct Service {
    session_component: Arc<SessionComponent>,
    sample: Arc<Mutex<SampleArgs>>,
    _manager: JoinHandle<()>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Device {
    Cpu,
    NvidiaGpu(i32),
}

impl Service {
    pub fn load_model(path: impl AsRef<Path>, sample: SampleArgs, device: Device) -> Self {
        let model_dir = path.as_ref().to_owned();
        let sample = Arc::new(Mutex::new(sample));
        let (sender, receiver) = channel();
        Service {
            session_component: Arc::new(SessionComponent {
                template: template(&model_dir),
                normalizer: normalizer(&model_dir),
                tokenizer: tokenizer(&model_dir),
                sender,
            }),
            sample: sample.clone(),
            _manager: thread::spawn(move || match device {
                Device::Cpu => cpu::task(model_dir, sample, receiver),
                #[cfg(detected_cuda)]
                Device::NvidiaGpu(n) => {
                    use transformer_nvidia::cuda;
                    cuda::init();
                    nvidia::task(cuda::Device::new(n), model_dir, sample, receiver);
                }
                #[cfg(not(detected_cuda))]
                _ => panic!("Unsupported device"),
            }),
        }
    }

    #[inline]
    pub fn launch(&self) -> Session {
        self.session_component.clone().into()
    }

    #[inline]
    pub fn sample_args(&self) -> SampleArgs {
        self.sample.lock().unwrap().clone()
    }

    #[inline]
    pub fn set_sample_args(&self, sample: SampleArgs) {
        *self.sample.lock().unwrap() = sample;
    }
}

enum Command {
    Infer {
        id: usize,
        prompt: Vec<utok>,
        responsing: Sender<utok>,
    },
    Drop {
        id: usize,
    },
}

fn template(model_dir: impl AsRef<Path>) -> Box<dyn Template + Send + Sync> {
    let path: String = model_dir.as_ref().display().to_string();
    let path = path.to_ascii_lowercase();
    if path.contains("tinyllama") {
        Box::new(template::ChatTinyLlama)
    } else {
        Box::new(template::ChatCPM)
    }
}

fn normalizer(model_dir: impl AsRef<Path>) -> Box<dyn Normalizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(_) => return Box::new(BPECommonNormalizer {}),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(_) => return Box::new(()),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenizer + Send + Sync> {
    use std::io::ErrorKind::NotFound;
    match BPE::from_model_file(model_dir.as_ref().join("tokenizer.model")) {
        Ok(bpe) => return Box::new(bpe),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match VocabTxt::from_txt_file(model_dir.as_ref().join("vocabs.txt")) {
        Ok(voc) => return Box::new(voc),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}
