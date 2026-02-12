use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// A project entry for the workspace picker (used in both sections).
#[derive(Debug, Clone)]
pub struct ProjectEntry {
    pub path: PathBuf,
    pub display_path: String, // ~/projects/foo style
    pub session_count: usize, // 0 if no .capseal
}

/// Workspace picker state — two flat lists, no tree.
pub struct PickerState {
    pub recent: Vec<ProjectEntry>,
    pub git_repos: Vec<ProjectEntry>,
    pub selected: usize,
    pub scroll_offset: usize,
    pub path_input_mode: bool,
    pub path_input: String,
}

impl PickerState {
    pub fn new() -> Self {
        let recent = find_capseal_projects();
        let recent_paths: HashSet<PathBuf> = recent.iter().map(|p| p.path.clone()).collect();
        let git_repos = find_git_repos(&recent_paths);

        Self {
            recent,
            git_repos,
            selected: 0,
            scroll_offset: 0,
            path_input_mode: false,
            path_input: String::new(),
        }
    }

    pub fn total_items(&self) -> usize {
        self.recent.len() + self.git_repos.len()
    }

    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn move_down(&mut self) {
        let max = self.total_items().saturating_sub(1);
        if self.selected < max {
            self.selected += 1;
        }
    }

    /// Get the path of the currently selected item.
    pub fn selected_path(&self) -> Option<&Path> {
        if self.selected < self.recent.len() {
            Some(&self.recent[self.selected].path)
        } else {
            let idx = self.selected - self.recent.len();
            self.git_repos.get(idx).map(|e| e.path.as_path())
        }
    }

    /// Keep selected item visible within viewport.
    pub fn adjust_scroll(&mut self, viewport_height: u16) {
        let vh = viewport_height as usize;
        // Reserve lines for header, section labels, footer
        // Header: 3 (title + blank + "Recent:"), footer: 3 (blank + hint + blank)
        // Between sections: 2 (blank + "Git repositories:")
        let overhead = if self.recent.is_empty() {
            6
        } else {
            8 + self.recent.len()
        };
        let visible_slots = vh.saturating_sub(overhead);

        if visible_slots == 0 {
            return;
        }

        // For scrolling, we care about the git_repos section since
        // recent is always visible. Scroll offset applies to git_repos.
        if self.selected < self.recent.len() {
            // In recent section — no scrolling needed
            self.scroll_offset = 0;
            return;
        }

        let git_idx = self.selected - self.recent.len();
        let margin = 2usize;

        if git_idx < self.scroll_offset.saturating_add(margin) {
            self.scroll_offset = git_idx.saturating_sub(margin);
        }
        if git_idx >= self.scroll_offset + visible_slots.saturating_sub(margin) {
            self.scroll_offset = git_idx.saturating_sub(visible_slots.saturating_sub(margin + 1));
        }
    }
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/// Find directories with .capseal/ in common locations. These are "Recent projects".
fn find_capseal_projects() -> Vec<ProjectEntry> {
    let mut projects = Vec::new();
    let Some(home) = dirs::home_dir() else {
        return projects;
    };

    let search_dirs = [
        home.join("projects"),
        home.join("code"),
        home.join("dev"),
        home.join("src"),
        home.join("repos"),
        home.join("workspace"),
        home.join("work"),
        home.clone(),
    ];

    for search_dir in &search_dirs {
        scan_one_level(search_dir, &home, &mut projects, true);
    }

    // Deduplicate by path, sort by session count descending
    projects.sort_by(|a, b| b.session_count.cmp(&a.session_count));
    projects.dedup_by(|a, b| a.path == b.path);
    projects.truncate(10);
    projects
}

/// Find directories with .git/ in common locations (max 2 levels deep).
/// Excludes paths already in `exclude`.
fn find_git_repos(exclude: &HashSet<PathBuf>) -> Vec<ProjectEntry> {
    let mut repos = Vec::new();
    let Some(home) = dirs::home_dir() else {
        return repos;
    };

    let search_dirs = [
        home.join("projects"),
        home.join("code"),
        home.join("dev"),
        home.join("src"),
        home.join("repos"),
        home.join("workspace"),
        home.join("work"),
        home.clone(),
    ];

    for search_dir in &search_dirs {
        // Level 1: direct children
        scan_one_level(search_dir, &home, &mut repos, false);
        // Level 2: grandchildren (e.g. ~/projects/foo/bar)
        if let Ok(entries) = std::fs::read_dir(search_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !name.starts_with('.') {
                        scan_one_level(&path, &home, &mut repos, false);
                    }
                }
            }
        }
    }

    // Remove entries already in recent
    repos.retain(|r| !exclude.contains(&r.path));

    // Deduplicate, sort alphabetically by display path
    repos.sort_by(|a, b| a.display_path.cmp(&b.display_path));
    repos.dedup_by(|a, b| a.path == b.path);
    repos
}

/// Scan one directory level for projects.
/// If `require_capseal` is true, only include dirs with .capseal/.
/// Otherwise, only include dirs with .git.
fn scan_one_level(dir: &Path, home: &Path, out: &mut Vec<ProjectEntry>, require_capseal: bool) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }

        let dominated = if require_capseal {
            path.join(".capseal").is_dir()
        } else {
            path.join(".git").exists()
        };

        if dominated {
            let session_count = if path.join(".capseal").is_dir() {
                count_sessions(&path.join(".capseal"))
            } else {
                0
            };
            out.push(ProjectEntry {
                display_path: tilde_path(&path, home),
                path,
                session_count,
            });
        }
    }
}

/// Count unique sessions in .capseal/runs/ by deduping run dirs and .cap files.
fn count_sessions(capseal_dir: &Path) -> usize {
    use std::collections::HashSet;

    let runs_dir = capseal_dir.join("runs");
    if !runs_dir.is_dir() {
        return 0;
    }

    std::fs::read_dir(&runs_dir)
        .map(|entries| {
            let mut session_ids: HashSet<String> = HashSet::new();
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if name == "latest" || name == "latest.cap" {
                    continue;
                }

                if path.is_dir() {
                    session_ids.insert(name);
                } else if name.ends_with(".cap") {
                    // Use .cap stem as canonical session key.
                    // This dedupes correctly against run directories even when
                    // manifest session_name differs.
                    let stem = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or(name);
                    session_ids.insert(stem);
                }
            }
            session_ids.len()
        })
        .unwrap_or(0)
}

/// Replace home prefix with ~
fn tilde_path(path: &Path, home: &Path) -> String {
    let s = path.display().to_string();
    let home_s = home.display().to_string();
    if s.starts_with(&home_s) {
        format!("~{}", &s[home_s.len()..])
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::count_sessions;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{ts}"))
    }

    #[test]
    fn dedupes_run_directory_and_matching_cap_stem() {
        let root = temp_dir("capseal-workspace-test");
        let capseal = root.join(".capseal");
        let runs = capseal.join("runs");
        fs::create_dir_all(runs.join("20260211T010101-mcp")).expect("create run dir");
        fs::write(runs.join("20260211T010101-mcp.cap"), b"dummy").expect("write cap");

        assert_eq!(count_sessions(&capseal), 1);

        let _ = fs::remove_dir_all(root);
    }
}
