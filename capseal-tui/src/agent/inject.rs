use anyhow::Result;
use std::path::Path;

/// Write or merge .mcp.json with CapSeal server configuration.
pub fn inject_mcp(workspace: &Path) -> Result<()> {
    let mcp_path = workspace.join(".mcp.json");

    let capseal_server = serde_json::json!({
        "command": "capseal",
        "args": ["mcp-serve", "-w", workspace.to_str().unwrap_or(".")],
        "type": "stdio"
    });

    let final_config = if mcp_path.exists() {
        // Merge with existing .mcp.json
        let content = std::fs::read_to_string(&mcp_path)?;
        let mut existing: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(servers) = existing
            .get_mut("mcpServers")
            .and_then(|s| s.as_object_mut())
        {
            servers.insert("capseal".to_string(), capseal_server);
        } else {
            existing["mcpServers"] = serde_json::json!({
                "capseal": capseal_server
            });
        }
        existing
    } else {
        serde_json::json!({
            "mcpServers": {
                "capseal": capseal_server
            }
        })
    };

    std::fs::write(&mcp_path, serde_json::to_string_pretty(&final_config)?)?;
    Ok(())
}
