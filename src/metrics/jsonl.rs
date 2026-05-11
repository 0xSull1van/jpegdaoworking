use super::Event;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;
use tokio::sync::mpsc;

pub async fn run_appender<P: AsRef<Path>>(
    path: P,
    mut rx: mpsc::Receiver<(SystemTime, Event)>,
) -> std::io::Result<()> {
    let path = path.as_ref().to_path_buf();
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let f = OpenOptions::new().create(true).append(true).open(&path)?;
    let mut w = std::io::BufWriter::new(f);
    while let Some((ts, ev)) = rx.recv().await {
        let ts_secs = ts
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let line = serde_json::json!({ "ts": ts_secs, "event": ev });
        writeln!(w, "{}", line)?;
        w.flush()?;
    }
    Ok(())
}
