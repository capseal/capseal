use bicep_io::cli::{Cli, Commands, run_sample_command};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Sample {
            model,
            calc,
            integrator,
            dt,
            steps,
            paths,
            save_stride,
            seed,
            out,
            params,
        } => {
            run_sample_command(
                model,
                calc, 
                integrator,
                dt,
                steps,
                paths,
                save_stride,
                seed,
                out,
                params,
            ).await?;
        }
    }
    
    Ok(())
}