use anyhow::Result;
use std::io::Read;
use std::path::Path;

/// Minimal reader for NumPy .npz files containing beta posterior parameters.
/// An .npz file is a ZIP archive of .npy files.
pub struct RiskModel {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl RiskModel {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        let alpha = read_npy_as_f64(&mut archive, "alpha.npy")
            .or_else(|_| read_npy_as_f64(&mut archive, "alphas.npy"))?;
        let beta = read_npy_as_f64(&mut archive, "beta.npy")
            .or_else(|_| read_npy_as_f64(&mut archive, "betas.npy"))?;

        Ok(Self { alpha, beta })
    }

    pub fn grid_size(&self) -> usize {
        self.alpha.len()
    }

    /// Compute p_fail for a grid index using Beta posterior mean.
    /// In capseal: alpha = failures, beta = successes.
    /// P(fail) = alpha / (alpha + beta)
    pub fn p_fail(&self, idx: usize) -> f64 {
        if idx >= self.alpha.len() {
            return 0.5;
        }
        let a = self.alpha[idx];
        let b = self.beta[idx];
        if a + b == 0.0 {
            return 0.5;
        }
        a / (a + b)
    }

    /// Get profiles with at least `min_obs` observations
    pub fn active_profiles(&self, min_obs: f64) -> Vec<(usize, f64)> {
        (0..self.grid_size())
            .filter(|&i| {
                let total = (self.alpha[i] - 1.0).max(0.0) + (self.beta[i] - 1.0).max(0.0);
                total >= min_obs
            })
            .map(|i| (i, self.p_fail(i)))
            .collect()
    }

    pub fn episode_count(&self) -> u32 {
        let total: f64 = self
            .alpha
            .iter()
            .zip(self.beta.iter())
            .map(|(a, b)| (a - 1.0).max(0.0) + (b - 1.0).max(0.0))
            .sum();
        total as u32
    }

    pub fn profile_count(&self) -> u32 {
        self.active_profiles(1.0).len() as u32
    }
}

/// Read a .npy file from inside a .npz archive, converting to f64.
/// Handles both float64 (<f8) and int64 (<i8) dtypes — Python's numpy
/// saves beta posteriors as int64 by default.
///
/// .npy format: magic (\x93NUMPY), version (2 bytes), header_len (2 or 4 bytes),
/// ASCII header dict, then raw data.
fn read_npy_as_f64(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<Vec<f64>> {
    let mut file = archive.by_name(name)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    // Parse .npy header
    // Magic: \x93NUMPY
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        anyhow::bail!("Invalid .npy magic for {}", name);
    }

    let major = buf[6];
    let _minor = buf[7];

    let (header_len, data_offset) = if major == 1 {
        let hl = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        (hl, 10 + hl)
    } else {
        // Version 2+: 4-byte header length
        let hl = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        (hl, 12 + hl)
    };

    // Parse header to get shape and dtype
    let header_str = std::str::from_utf8(&buf[data_offset - header_len..data_offset])?;
    let num_elements = parse_shape(header_str)?;
    let descr = parse_descr(header_str)?;

    let data_bytes = &buf[data_offset..];

    match descr.as_str() {
        "<f8" | "=f8" | "|f8" => {
            // float64: 8 bytes each, read directly
            let expected = num_elements * 8;
            if data_bytes.len() < expected {
                anyhow::bail!("Not enough data for {} f64s in {}", num_elements, name);
            }
            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                let off = i * 8;
                let bytes: [u8; 8] = data_bytes[off..off + 8].try_into()?;
                values.push(f64::from_le_bytes(bytes));
            }
            Ok(values)
        }
        "<i8" | "=i8" | "|i8" => {
            // int64: 8 bytes each, cast to f64
            let expected = num_elements * 8;
            if data_bytes.len() < expected {
                anyhow::bail!("Not enough data for {} i64s in {}", num_elements, name);
            }
            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                let off = i * 8;
                let bytes: [u8; 8] = data_bytes[off..off + 8].try_into()?;
                values.push(i64::from_le_bytes(bytes) as f64);
            }
            Ok(values)
        }
        "<f4" | "=f4" | "|f4" => {
            // float32: 4 bytes each, cast to f64
            let expected = num_elements * 4;
            if data_bytes.len() < expected {
                anyhow::bail!("Not enough data for {} f32s in {}", num_elements, name);
            }
            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                let off = i * 4;
                let bytes: [u8; 4] = data_bytes[off..off + 4].try_into()?;
                values.push(f32::from_le_bytes(bytes) as f64);
            }
            Ok(values)
        }
        "<i4" | "=i4" | "|i4" => {
            // int32: 4 bytes each, cast to f64
            let expected = num_elements * 4;
            if data_bytes.len() < expected {
                anyhow::bail!("Not enough data for {} i32s in {}", num_elements, name);
            }
            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                let off = i * 4;
                let bytes: [u8; 4] = data_bytes[off..off + 4].try_into()?;
                values.push(i32::from_le_bytes(bytes) as f64);
            }
            Ok(values)
        }
        "<u8" | "=u8" | "|u8" => {
            // uint8: 1 byte each, cast to f64
            if data_bytes.len() < num_elements {
                anyhow::bail!("Not enough data for {} u8s in {}", num_elements, name);
            }
            Ok(data_bytes[..num_elements]
                .iter()
                .map(|&b| b as f64)
                .collect())
        }
        _ => {
            anyhow::bail!("Unsupported numpy dtype '{}' in {}", descr, name);
        }
    }
}

/// Extract the 'descr' field from a .npy header dict string.
/// Example header: "{'descr': '<i8', 'fortran_order': False, 'shape': (256,), }"
fn parse_descr(header: &str) -> Result<String> {
    // Look for 'descr': '<XX'  or  'descr': '|XX'
    if let Some(idx) = header.find("'descr'") {
        let rest = &header[idx + 7..]; // skip "'descr'"
                                       // Find the opening quote of the value
        if let Some(q1) = rest.find('\'') {
            let after_q1 = &rest[q1 + 1..];
            if let Some(q2) = after_q1.find('\'') {
                return Ok(after_q1[..q2].to_string());
            }
        }
    }
    // Default to float64 if descr not found (backwards compat)
    Ok("<f8".to_string())
}

fn parse_shape(header: &str) -> Result<usize> {
    // Find 'shape': (N,) or 'shape': (N, M) or 'shape': ()
    if let Some(idx) = header.find("'shape'") {
        let rest = &header[idx..];
        if let Some(paren_start) = rest.find('(') {
            if let Some(paren_end) = rest[paren_start..].find(')') {
                let shape_str = &rest[paren_start + 1..paren_start + paren_end];
                let shape_str = shape_str.trim();
                if shape_str.is_empty() {
                    return Ok(1); // scalar
                }
                // Parse comma-separated dimensions
                let total: usize = shape_str
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .product();
                return Ok(total.max(1));
            }
        }
    }
    anyhow::bail!("Could not parse shape from npy header: {}", header);
}
