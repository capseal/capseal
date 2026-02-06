// Independent Verifier - Go implementation
//
// This is a DELIBERATELY INDEPENDENT implementation of the verification logic.
// It shares NO code with the Python implementation to catch correlated bugs.
//
// Implements:
// - Canonical row encoding
// - SHA256 digest of row
// - Hash chain evolution (head_{t+1} = H(head_t || ":" || d_t))
// - Merkle root + path verification
// - Sidecar hash verification
//
// Usage:
//   go run verifier.go verify-trace <trace.jsonl> <commitments.json>
//   go run verifier.go verify-sidecar <features.csv> <sidecar.json>
//   go run verifier.go verify-opening <opening.json> [checkpoint.json]
//   go run verifier.go verify-merkle <leaf> <proof.json> <root>

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// =============================================================================
// CANONICAL JSON SERIALIZATION
// =============================================================================

// canonicalJSON produces deterministic JSON (sorted keys, no whitespace)
func canonicalJSON(v interface{}) ([]byte, error) {
	// First marshal to get the structure
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	// Unmarshal to interface{} and re-marshal with sorted keys
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, err
	}

	return marshalCanonical(obj)
}

func marshalCanonical(v interface{}) ([]byte, error) {
	switch val := v.(type) {
	case map[string]interface{}:
		// Sort keys
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		var buf strings.Builder
		buf.WriteString("{")
		for i, k := range keys {
			if i > 0 {
				buf.WriteString(",")
			}
			keyBytes, _ := json.Marshal(k)
			buf.Write(keyBytes)
			buf.WriteString(":")
			valBytes, err := marshalCanonical(val[k])
			if err != nil {
				return nil, err
			}
			buf.Write(valBytes)
		}
		buf.WriteString("}")
		return []byte(buf.String()), nil

	case []interface{}:
		var buf strings.Builder
		buf.WriteString("[")
		for i, item := range val {
			if i > 0 {
				buf.WriteString(",")
			}
			itemBytes, err := marshalCanonical(item)
			if err != nil {
				return nil, err
			}
			buf.Write(itemBytes)
		}
		buf.WriteString("]")
		return []byte(buf.String()), nil

	default:
		return json.Marshal(v)
	}
}

func hashCanonical(v interface{}) (string, error) {
	data, err := canonicalJSON(v)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// =============================================================================
// TRACE ROW
// =============================================================================

type TraceRow struct {
	Schema    string                   `json:"schema"`
	T         int                      `json:"t"`
	RandAddrs []map[string]interface{} `json:"rand_addrs"`
	ViewPre   map[string]interface{}   `json:"view_pre"`
	ViewPost  map[string]interface{}   `json:"view_post"`
	XT        []float64                `json:"x_t"`
	Aux       map[string]interface{}   `json:"aux"`
}

func (r *TraceRow) ComputeDigest() (string, error) {
	// Convert to map for canonical serialization
	rowMap := map[string]interface{}{
		"schema":     r.Schema,
		"t":          r.T,
		"rand_addrs": r.RandAddrs,
		"view_pre":   r.ViewPre,
		"view_post":  r.ViewPost,
		"x_t":        r.XT,
		"aux":        r.Aux,
	}
	return hashCanonical(rowMap)
}

// =============================================================================
// HASH CHAIN
// =============================================================================

type HashChainState struct {
	Head    string
	Step    int
	Digests []string
}

func GenesisChain(manifestHash string) *HashChainState {
	// head_0 = H("genesis:" || manifest_hash)
	preimage := "genesis:" + manifestHash
	hash := sha256.Sum256([]byte(preimage))
	return &HashChainState{
		Head:    hex.EncodeToString(hash[:]),
		Step:    0,
		Digests: []string{},
	}
}

func (c *HashChainState) Append(row *TraceRow) (string, error) {
	// d_t = H(row_t)
	dt, err := row.ComputeDigest()
	if err != nil {
		return "", err
	}
	c.Digests = append(c.Digests, dt)

	// head_{t+1} = H(head_t || ":" || d_t)
	preimage := c.Head + ":" + dt
	hash := sha256.Sum256([]byte(preimage))
	c.Head = hex.EncodeToString(hash[:])
	c.Step++

	return dt, nil
}

// =============================================================================
// MERKLE TREE
// =============================================================================

func nextPowerOf2(n int) int {
	size := 1
	for size < n {
		size *= 2
	}
	return size
}

func ComputeMerkleRoot(digests []string) string {
	if len(digests) == 0 {
		hash := sha256.Sum256([]byte("empty"))
		return hex.EncodeToString(hash[:])
	}

	// Pad to power of 2
	size := nextPowerOf2(len(digests))
	leaves := make([]string, size)
	copy(leaves, digests)
	for i := len(digests); i < size; i++ {
		leaves[i] = strings.Repeat("0", 64)
	}

	// Build tree bottom-up
	for len(leaves) > 1 {
		nextLevel := make([]string, len(leaves)/2)
		for i := 0; i < len(leaves); i += 2 {
			combined := leaves[i] + ":" + leaves[i+1]
			hash := sha256.Sum256([]byte(combined))
			nextLevel[i/2] = hex.EncodeToString(hash[:])
		}
		leaves = nextLevel
	}

	return leaves[0]
}

type MerkleProofElement struct {
	Sibling   string `json:"sibling"`
	Direction string `json:"direction"` // "L" or "R"
}

func VerifyMerkleProof(leafDigest string, proof []MerkleProofElement, root string) bool {
	current := leafDigest
	for _, elem := range proof {
		var combined string
		if elem.Direction == "L" {
			combined = elem.Sibling + ":" + current
		} else {
			combined = current + ":" + elem.Sibling
		}
		hash := sha256.Sum256([]byte(combined))
		current = hex.EncodeToString(hash[:])
	}
	return current == root
}

// =============================================================================
// FILE HASH
// =============================================================================

func ComputeFileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// =============================================================================
// STRUCTURES
// =============================================================================

type Commitments struct {
	ManifestHash    string `json:"manifest_hash"`
	HeadT           string `json:"head_T"`
	TotalSteps      int    `json:"total_steps"`
	TotalCheckpoints int   `json:"total_checkpoints"`
}

type Sidecar struct {
	Schema             string `json:"schema"`
	FeaturesShardHash  string `json:"features_shard_hash"`
	FeaturesRowCount   int    `json:"features_row_count"`
	FeaturesDim        int    `json:"features_dim"`
	TraceAnchorType    string `json:"trace_anchor_type"`
	CheckpointIndex    int    `json:"checkpoint_index"`
	StepStart          int    `json:"step_start"`
	StepEnd            int    `json:"step_end"`
	HeadAtEnd          string `json:"head_at_end"`
	ManifestHash       string `json:"manifest_hash"`
	PolicyHash         string `json:"policy_hash"`
	InputsHash         string `json:"inputs_hash"`
	CreatedAt          string `json:"created_at"`
	BicepVersion       string `json:"bicep_version"`
	SidecarHash        string `json:"sidecar_hash"`
}

type RowOpening struct {
	Row             map[string]interface{}   `json:"row"`
	MerkleProof     [][]interface{}          `json:"merkle_proof"`
	LeafIndex       int                      `json:"leaf_index"`
	CheckpointIndex int                      `json:"checkpoint_index"`
	ChunkRoot       string                   `json:"chunk_root"`
	StepStart       int                      `json:"step_start"`
	StepEnd         int                      `json:"step_end"`
	HeadAtStart     string                   `json:"head_at_start"`
	HeadAtEnd       string                   `json:"head_at_end"`
	ManifestHash    string                   `json:"manifest_hash"`
}

type CheckpointReceipt struct {
	Schema          string   `json:"schema"`
	ManifestHash    string   `json:"manifest_hash"`
	CheckpointIndex int      `json:"checkpoint_index"`
	StepStart       int      `json:"step_start"`
	StepEnd         int      `json:"step_end"`
	HeadAtStart     string   `json:"head_at_start"`
	HeadAtEnd       string   `json:"head_at_end"`
	ChunkRoot       string   `json:"chunk_root"`
	RowDigests      []string `json:"row_digests"`
	OutputsHash     string   `json:"outputs_hash"`
	ReceiptHash     string   `json:"receipt_hash"`
}

// =============================================================================
// VERIFY TRACE
// =============================================================================

func verifyTrace(tracePath, commitmentsPath string) error {
	// Load commitments
	commitmentsData, err := os.ReadFile(commitmentsPath)
	if err != nil {
		return fmt.Errorf("failed to read commitments: %w", err)
	}

	var commitments Commitments
	if err := json.Unmarshal(commitmentsData, &commitments); err != nil {
		return fmt.Errorf("failed to parse commitments: %w", err)
	}

	// Replay hash chain
	chain := GenesisChain(commitments.ManifestHash)

	traceFile, err := os.Open(tracePath)
	if err != nil {
		return fmt.Errorf("failed to open trace: %w", err)
	}
	defer traceFile.Close()

	scanner := bufio.NewScanner(traceFile)
	stepCount := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var row TraceRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return fmt.Errorf("failed to parse row %d: %w", stepCount, err)
		}

		if _, err := chain.Append(&row); err != nil {
			return fmt.Errorf("failed to append row %d: %w", stepCount, err)
		}
		stepCount++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read trace: %w", err)
	}

	// Verify
	if stepCount != commitments.TotalSteps {
		return fmt.Errorf("step count mismatch: got %d, expected %d", stepCount, commitments.TotalSteps)
	}

	if chain.Head != commitments.HeadT {
		return fmt.Errorf("head mismatch:\n  got:      %s\n  expected: %s", chain.Head, commitments.HeadT)
	}

	fmt.Printf("[GO-VERIFIER] PASS: Verified %d steps, head matches\n", stepCount)
	return nil
}

// =============================================================================
// VERIFY SIDECAR
// =============================================================================

func verifySidecar(featuresPath, sidecarPath string) error {
	// Load sidecar
	sidecarData, err := os.ReadFile(sidecarPath)
	if err != nil {
		return fmt.Errorf("failed to read sidecar: %w", err)
	}

	var sidecar Sidecar
	if err := json.Unmarshal(sidecarData, &sidecar); err != nil {
		return fmt.Errorf("failed to parse sidecar: %w", err)
	}

	// Compute features hash
	actualHash, err := ComputeFileHash(featuresPath)
	if err != nil {
		return fmt.Errorf("failed to hash features: %w", err)
	}

	// Compare
	if actualHash != sidecar.FeaturesShardHash {
		return fmt.Errorf("features hash mismatch:\n  actual:   %s\n  expected: %s", actualHash, sidecar.FeaturesShardHash)
	}

	fmt.Printf("[GO-VERIFIER] PASS: Features hash verified (%s...)\n", actualHash[:16])
	return nil
}

// =============================================================================
// VERIFY OPENING
// =============================================================================

func verifyOpening(openingPath string, checkpointPath string) error {
	// Load opening
	openingData, err := os.ReadFile(openingPath)
	if err != nil {
		return fmt.Errorf("failed to read opening: %w", err)
	}

	var opening RowOpening
	if err := json.Unmarshal(openingData, &opening); err != nil {
		return fmt.Errorf("failed to parse opening: %w", err)
	}

	// Convert row to TraceRow and compute digest
	rowDigest, err := hashCanonical(opening.Row)
	if err != nil {
		return fmt.Errorf("failed to compute row digest: %w", err)
	}

	// Convert merkle proof
	proof := make([]MerkleProofElement, len(opening.MerkleProof))
	for i, p := range opening.MerkleProof {
		if len(p) != 2 {
			return fmt.Errorf("invalid merkle proof element at index %d", i)
		}
		sibling, ok1 := p[0].(string)
		direction, ok2 := p[1].(string)
		if !ok1 || !ok2 {
			return fmt.Errorf("invalid merkle proof element types at index %d", i)
		}
		proof[i] = MerkleProofElement{Sibling: sibling, Direction: direction}
	}

	// Verify merkle proof
	if !VerifyMerkleProof(rowDigest, proof, opening.ChunkRoot) {
		return fmt.Errorf("merkle proof invalid for row t=%v", opening.Row["t"])
	}

	// Cross-check with checkpoint if provided
	if checkpointPath != "" {
		checkpointData, err := os.ReadFile(checkpointPath)
		if err != nil {
			return fmt.Errorf("failed to read checkpoint: %w", err)
		}

		var checkpoint CheckpointReceipt
		if err := json.Unmarshal(checkpointData, &checkpoint); err != nil {
			return fmt.Errorf("failed to parse checkpoint: %w", err)
		}

		if opening.ChunkRoot != checkpoint.ChunkRoot {
			return fmt.Errorf("chunk root mismatch with checkpoint")
		}

		if opening.ManifestHash != checkpoint.ManifestHash {
			return fmt.Errorf("manifest hash mismatch with checkpoint")
		}
	}

	fmt.Printf("[GO-VERIFIER] PASS: Row opening verified (t=%v, leaf_index=%d)\n", opening.Row["t"], opening.LeafIndex)
	return nil
}

// =============================================================================
// VERIFY MERKLE (standalone)
// =============================================================================

func verifyMerkle(leaf, proofPath, root string) error {
	proofData, err := os.ReadFile(proofPath)
	if err != nil {
		return fmt.Errorf("failed to read proof: %w", err)
	}

	var rawProof [][]string
	if err := json.Unmarshal(proofData, &rawProof); err != nil {
		return fmt.Errorf("failed to parse proof: %w", err)
	}

	proof := make([]MerkleProofElement, len(rawProof))
	for i, p := range rawProof {
		if len(p) != 2 {
			return fmt.Errorf("invalid proof element at index %d", i)
		}
		proof[i] = MerkleProofElement{Sibling: p[0], Direction: p[1]}
	}

	if !VerifyMerkleProof(leaf, proof, root) {
		return fmt.Errorf("merkle proof invalid")
	}

	fmt.Printf("[GO-VERIFIER] PASS: Merkle proof verified\n")
	return nil
}

// =============================================================================
// MAIN
// =============================================================================

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage:")
		fmt.Println("  go run verifier.go verify-trace <trace.jsonl> <commitments.json>")
		fmt.Println("  go run verifier.go verify-sidecar <features.csv> <sidecar.json>")
		fmt.Println("  go run verifier.go verify-opening <opening.json> [checkpoint.json]")
		fmt.Println("  go run verifier.go verify-merkle <leaf> <proof.json> <root>")
		os.Exit(1)
	}

	cmd := os.Args[1]
	var err error

	switch cmd {
	case "verify-trace":
		if len(os.Args) != 4 {
			fmt.Println("Usage: verify-trace <trace.jsonl> <commitments.json>")
			os.Exit(1)
		}
		err = verifyTrace(os.Args[2], os.Args[3])

	case "verify-sidecar":
		if len(os.Args) != 4 {
			fmt.Println("Usage: verify-sidecar <features.csv> <sidecar.json>")
			os.Exit(1)
		}
		err = verifySidecar(os.Args[2], os.Args[3])

	case "verify-opening":
		if len(os.Args) < 3 {
			fmt.Println("Usage: verify-opening <opening.json> [checkpoint.json]")
			os.Exit(1)
		}
		checkpointPath := ""
		if len(os.Args) >= 4 {
			checkpointPath = os.Args[3]
		}
		err = verifyOpening(os.Args[2], checkpointPath)

	case "verify-merkle":
		if len(os.Args) != 5 {
			fmt.Println("Usage: verify-merkle <leaf> <proof.json> <root>")
			os.Exit(1)
		}
		err = verifyMerkle(os.Args[2], os.Args[3], os.Args[4])

	default:
		fmt.Printf("Unknown command: %s\n", cmd)
		os.Exit(1)
	}

	if err != nil {
		fmt.Printf("[GO-VERIFIER] FAIL: %v\n", err)
		os.Exit(1)
	}
}
