mod config;
mod indexer;
mod query;
mod storage;
mod tools;

use rmcp::{ServiceExt, transport::stdio};
use tools::Yomu;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("yomu=info".parse()?),
        )
        .init();

    info!("starting yomu MCP server");

    let service = Yomu::new()?
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!("failed to start server: {e}"))?;

    // SQLite WAL mode handles crash recovery — in-flight transactions are
    // rolled back and -wal/-shm files are cleaned up on next open_db() call.
    tokio::select! {
        result = service.waiting() => { let _ = result?; }
        _ = tokio::signal::ctrl_c() => {
            info!("received shutdown signal, stopping");
        }
    }
    info!("server stopped");
    Ok(())
}
