
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum FrinZError {
    Io(io::Error),
    
    // Add other custom error types as needed
}

impl fmt::Display for FrinZError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FrinZError::Io(ref err) => write!(f, "IO error: {}", err),
        }
    }
}

impl From<io::Error> for FrinZError {
    fn from(err: io::Error) -> FrinZError {
        FrinZError::Io(err)
    }
}
