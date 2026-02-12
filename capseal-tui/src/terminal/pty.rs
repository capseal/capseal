use anyhow::Result;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::io::{Read, Write};
use std::path::Path;

pub struct PtyHandle {
    master: Box<dyn portable_pty::MasterPty + Send>,
    child: Box<dyn portable_pty::Child + Send>,
    reader: Box<dyn Read + Send>,
    writer: Box<dyn Write + Send>,
}

impl PtyHandle {
    pub fn spawn(cwd: &Path, cols: u16, rows: u16) -> Result<Self> {
        let pty_system = native_pty_system();

        let pair = pty_system.openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;

        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
        let is_bash = shell.ends_with("bash");
        let is_zsh = shell.ends_with("zsh");

        let mut cmd = if is_bash {
            let mut c = CommandBuilder::new(&shell);
            c.arg("--norc");
            c.arg("--noprofile");
            c
        } else {
            CommandBuilder::new(&shell)
        };

        cmd.cwd(cwd);
        cmd.env("TERM", "xterm-256color");
        cmd.env("CAPSEAL_ACTIVE", "1");
        cmd.env("CAPSEAL_TUI", "1");
        cmd.env("CAPSEAL_WORKSPACE", cwd.to_str().unwrap_or("."));
        cmd.env("PROMPT_TOOLKIT_NO_CPR", "1");
        cmd.env("PS1", "\u{276f} ");

        // For zsh: point ZDOTDIR to a temp dir with minimal .zshrc
        if is_zsh {
            let empty_zsh_dir = std::path::PathBuf::from("/tmp/capseal-empty-zsh");
            if !empty_zsh_dir.exists() {
                let _ = std::fs::create_dir_all(&empty_zsh_dir);
            }
            let zshrc = empty_zsh_dir.join(".zshrc");
            if !zshrc.exists() {
                let _ = std::fs::write(&zshrc, "PS1='\u{276f} '\nunsetopt PROMPT_SUBST\n");
            }
            cmd.env(
                "ZDOTDIR",
                empty_zsh_dir.to_str().unwrap_or("/tmp/capseal-empty-zsh"),
            );
        }

        let child = pair.slave.spawn_command(cmd)?;

        // Drop slave — master communicates via fd
        drop(pair.slave);

        let reader = pair.master.try_clone_reader()?;
        let writer = pair.master.take_writer()?;

        Ok(Self {
            master: pair.master,
            child,
            reader,
            writer,
        })
    }

    pub fn resize(&self, cols: u16, rows: u16) -> Result<()> {
        self.master.resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        Ok(())
    }

    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        self.writer.flush()?;
        Ok(())
    }

    pub fn is_alive(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_none()
    }

    /// Get a reference to the reader for use in a background thread
    pub fn take_reader(&mut self) -> Box<dyn Read + Send> {
        // Replace with a dummy reader — caller takes ownership
        std::mem::replace(&mut self.reader, Box::new(std::io::empty()))
    }
}
