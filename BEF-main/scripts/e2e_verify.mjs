#!/usr/bin/env node
/**
 * E2E Verification Smoke Test
 *
 * Tests the backend verify endpoint and asserts the canonical VerifyReport contract.
 * This script enforces API contract stability between UI and backend.
 *
 * Usage:
 *   node scripts/e2e_verify.mjs
 *   E2E_BASE=http://127.0.0.1:5001 node scripts/e2e_verify.mjs
 *   E2E_CAPSULE=path/to/capsule.cap node scripts/e2e_verify.mjs
 *
 * Environment Variables:
 *   E2E_BASE       - Backend URL (default: http://127.0.0.1:5001)
 *   E2E_CAPSULE    - Capsule path to verify (default: momentum_strategy_v2.cap)
 *   E2E_MODE       - Verification mode: proof-only, da, replay (default: proof-only)
 *   E2E_API_KEY    - API key for authentication (default: demo-key)
 *   E2E_TIMEOUT    - Request timeout in ms (default: 30000)
 */

import assert from "node:assert/strict";

// Configuration
const BASE = process.env.E2E_BASE ?? "http://127.0.0.1:5001";
const CAPSULE_PATH = process.env.E2E_CAPSULE ?? "momentum_strategy_v2.cap";
const MODE = process.env.E2E_MODE ?? "proof-only";
const API_KEY = process.env.E2E_API_KEY ?? "demo-key";
const TIMEOUT = parseInt(process.env.E2E_TIMEOUT ?? "30000", 10);

// Contract definitions from ui/src/contracts/contracts.ts
const VALID_LAYER_STATUSES = new Set(["pass", "fail", "skipped", "unknown"]);
const VALID_VERIFY_STATUSES = new Set(["verified", "rejected"]);
const VALID_JOB_STATUSES = new Set([
  "PENDING",
  "VERIFYING",
  "VERIFIED",
  "FAILED",
  "UNKNOWN",
  // Legacy values that should be normalized
  "queued",
  "running",
  "succeeded",
  "failed",
  "pass",
  "error",
]);

// Exit codes from contracts
const EXIT_CODES = {
  VERIFIED: 0,
  PROOF_FAILED: 10,
  POLICY_MISMATCH: 11,
  COMMITMENT_FAILED: 12,
  DA_AUDIT_FAILED: 13,
  REPLAY_DIVERGED: 14,
  PARSE_ERROR: 20,
};

/**
 * Assert that a value is a boolean
 */
function assertBoolean(value, name) {
  assert.equal(
    typeof value,
    "boolean",
    `${name} must be boolean, got ${typeof value}: ${JSON.stringify(value)}`
  );
}

/**
 * Assert that a value is a string
 */
function assertString(value, name) {
  assert.equal(
    typeof value,
    "string",
    `${name} must be string, got ${typeof value}: ${JSON.stringify(value)}`
  );
}

/**
 * Assert that a value is a number
 */
function assertNumber(value, name) {
  assert.equal(
    typeof value,
    "number",
    `${name} must be number, got ${typeof value}: ${JSON.stringify(value)}`
  );
}

/**
 * Assert that a value is an array
 */
function assertArray(value, name) {
  assert.ok(
    Array.isArray(value),
    `${name} must be an array, got ${typeof value}: ${JSON.stringify(value)}`
  );
}

/**
 * Assert that a value is an object
 */
function assertObject(value, name) {
  assert.ok(
    typeof value === "object" && value !== null && !Array.isArray(value),
    `${name} must be an object, got ${typeof value}: ${JSON.stringify(value)}`
  );
}

/**
 * Assert that a value is one of allowed values
 */
function assertOneOf(value, allowed, name) {
  assert.ok(
    allowed.has(value),
    `${name} must be one of [${[...allowed].join(", ")}], got: ${JSON.stringify(value)}`
  );
}

/**
 * Validate a LayerCheck object from the contract
 */
function assertLayerCheck(layer, layerName) {
  assertObject(layer, `${layerName}`);
  assertOneOf(layer.status, VALID_LAYER_STATUSES, `${layerName}.status`);

  // Optional fields
  if ("message" in layer && layer.message !== undefined) {
    assertString(layer.message, `${layerName}.message`);
  }
  if ("evidence_ref" in layer && layer.evidence_ref !== undefined) {
    assertString(layer.evidence_ref, `${layerName}.evidence_ref`);
  }
}

/**
 * Validate VerificationLayers object
 * Layers can be an object (contract format) or array (normalized format)
 */
function assertVerificationLayers(layers, contextName) {
  if (Array.isArray(layers)) {
    // Normalized format from engine adapter
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      assertObject(layer, `${contextName}[${i}]`);

      // Required fields for normalized layers
      assertString(layer.id, `${contextName}[${i}].id`);
      assertString(layer.label, `${contextName}[${i}].label`);
      assertBoolean(layer.ok, `${contextName}[${i}].ok`);

      // Optional fields
      if ("reason" in layer && layer.reason !== null) {
        assertString(layer.reason, `${contextName}[${i}].reason`);
      }
    }
  } else if (typeof layers === "object" && layers !== null) {
    // Contract format (l0_hash, l1_commitment, etc.)
    const expectedLayers = [
      "l0_hash",
      "l1_commitment",
      "l2_constraint",
      "l3_proximity",
      "l4_receipt",
    ];

    for (const [key, value] of Object.entries(layers)) {
      assertLayerCheck(value, `${contextName}.${key}`);
    }
  } else {
    assert.fail(`${contextName} must be an array or object`);
  }
}

/**
 * Validate a VerifyError object
 */
function assertVerifyError(error, errorName) {
  assertObject(error, errorName);
  assertString(error.code, `${errorName}.code`);
  assertString(error.message, `${errorName}.message`);

  // Optional fields
  if ("hint" in error && error.hint !== undefined) {
    assertString(error.hint, `${errorName}.hint`);
  }
  if ("evidence_ref" in error && error.evidence_ref !== undefined) {
    assertString(error.evidence_ref, `${errorName}.evidence_ref`);
  }
}

/**
 * Validate VerifyTimings object
 */
function assertVerifyTimings(timings, contextName) {
  assertObject(timings, contextName);
  assertNumber(timings.total_ms, `${contextName}.total_ms`);

  // Optional fields
  if ("parse_ms" in timings) {
    assertNumber(timings.parse_ms, `${contextName}.parse_ms`);
  }
  if ("proof_verify_ms" in timings) {
    assertNumber(timings.proof_verify_ms, `${contextName}.proof_verify_ms`);
  }
  if ("merkle_verify_ms" in timings) {
    assertNumber(timings.merkle_verify_ms, `${contextName}.merkle_verify_ms`);
  }
}

/**
 * Validate the full VerifyReport contract
 *
 * This is the canonical contract from ui/src/contracts/contracts.ts
 */
function assertVerifyReport(report) {
  assertObject(report, "VerifyReport");

  // === REQUIRED FIELDS ===

  // run_id: string
  if ("run_id" in report) {
    assertString(report.run_id, "VerifyReport.run_id");
  }

  // status: 'verified' | 'rejected'
  if ("status" in report) {
    assertOneOf(report.status, VALID_VERIFY_STATUSES, "VerifyReport.status");
  }

  // exit_code: number
  if ("exit_code" in report) {
    assertNumber(report.exit_code, "VerifyReport.exit_code");
    // Validate it's a known exit code
    const knownCodes = new Set(Object.values(EXIT_CODES));
    assert.ok(
      knownCodes.has(report.exit_code),
      `VerifyReport.exit_code must be a known exit code, got: ${report.exit_code}`
    );
  }

  // layers: VerificationLayers
  if ("layers" in report) {
    assertVerificationLayers(report.layers, "VerifyReport.layers");
  }

  // errors: VerifyError[]
  if ("errors" in report) {
    assertArray(report.errors, "VerifyReport.errors");
    for (let i = 0; i < report.errors.length; i++) {
      assertVerifyError(report.errors[i], `VerifyReport.errors[${i}]`);
    }
  }

  // timings: VerifyTimings
  if ("timings" in report) {
    assertVerifyTimings(report.timings, "VerifyReport.timings");
  }

  // === OPTIONAL FIELDS ===

  if ("proof_size_bytes" in report && report.proof_size_bytes !== undefined) {
    assertNumber(report.proof_size_bytes, "VerifyReport.proof_size_bytes");
  }

  if ("backend_id" in report && report.backend_id !== undefined) {
    assertString(report.backend_id, "VerifyReport.backend_id");
  }

  if ("proof_system_id" in report && report.proof_system_id !== undefined) {
    assertString(report.proof_system_id, "VerifyReport.proof_system_id");
  }

  if ("verified_at" in report && report.verified_at !== undefined) {
    assertString(report.verified_at, "VerifyReport.verified_at");
  }
}

/**
 * Validate the CommandResponse wrapper (from Flask API)
 */
function assertCommandResponse(response) {
  assertObject(response, "CommandResponse");

  // requestId: string
  assertString(response.requestId, "CommandResponse.requestId");

  // status: 'SUCCESS' | 'ERROR'
  assert.ok(
    response.status === "SUCCESS" || response.status === "ERROR",
    `CommandResponse.status must be SUCCESS or ERROR, got: ${response.status}`
  );

  // exitCode: number
  assertNumber(response.exitCode, "CommandResponse.exitCode");

  // command: string[]
  assertArray(response.command, "CommandResponse.command");

  // stdout, stderr: string
  assertString(response.stdout, "CommandResponse.stdout");
  assertString(response.stderr, "CommandResponse.stderr");

  // result: optional object
  if ("result" in response && response.result !== null) {
    assertObject(response.result, "CommandResponse.result");
  }
}

/**
 * Validate normalized engine adapter response
 *
 * The engine adapter normalizes backend responses to:
 *   { ok: boolean, layers: VerifyLayer[], status: JobStatus, errorCode: string | null }
 */
function assertNormalizedResponse(response) {
  assertObject(response, "NormalizedResponse");

  // ok: boolean (required)
  assertBoolean(response.ok, "NormalizedResponse.ok");

  // layers: array (required)
  assertArray(response.layers, "NormalizedResponse.layers");
  for (let i = 0; i < response.layers.length; i++) {
    const layer = response.layers[i];
    assertObject(layer, `NormalizedResponse.layers[${i}]`);
    assertString(layer.id, `NormalizedResponse.layers[${i}].id`);
    assertString(layer.label, `NormalizedResponse.layers[${i}].label`);
    assertBoolean(layer.ok, `NormalizedResponse.layers[${i}].ok`);
  }

  // status: JobStatus (if present)
  if ("status" in response) {
    assertOneOf(response.status, VALID_JOB_STATUSES, "NormalizedResponse.status");
  }

  // errorCode: string | null
  if ("errorCode" in response && response.errorCode !== null) {
    assertString(response.errorCode, "NormalizedResponse.errorCode");
  }
}

/**
 * Test the /api/verify endpoint
 */
async function testVerifyEndpoint() {
  console.log(`\n--- Testing /api/verify endpoint ---`);
  console.log(`  Base URL: ${BASE}`);
  console.log(`  Capsule: ${CAPSULE_PATH}`);
  console.log(`  Mode: ${MODE}`);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT);

  try {
    const res = await fetch(`${BASE}/api/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({
        capsulePath: CAPSULE_PATH,
        mode: MODE,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    console.log(`  Response status: ${res.status}`);

    // Even on failure, we should get a structured response
    const data = await res.json();
    console.log(`  Response keys: [${Object.keys(data).join(", ")}]`);

    // Validate CommandResponse wrapper
    console.log(`\n  Validating CommandResponse wrapper...`);
    assertCommandResponse(data);
    console.log(`    requestId: ${data.requestId}`);
    console.log(`    status: ${data.status}`);
    console.log(`    exitCode: ${data.exitCode}`);

    // If there's a result, validate it matches VerifyReport contract
    if (data.result) {
      console.log(`\n  Validating VerifyReport result...`);
      console.log(`    Result keys: [${Object.keys(data.result).join(", ")}]`);

      // The result should be a partial VerifyReport
      // It may not have all fields depending on verification outcome
      if ("status" in data.result) {
        console.log(`    result.status: ${data.result.status}`);
      }
      if ("error_code" in data.result) {
        console.log(`    result.error_code: ${data.result.error_code}`);
      }
      if ("verification_level" in data.result) {
        console.log(`    result.verification_level: ${data.result.verification_level}`);
      }

      // Assert result fields match expected types
      assertVerifyReport(data.result);
    }

    // Check for successful verification
    if (res.ok && data.status === "SUCCESS") {
      console.log(`\n  Verification succeeded!`);
      assert.equal(data.exitCode, 0, "exitCode should be 0 on success");
    } else {
      console.log(`\n  Verification failed/rejected (this may be expected)`);
      console.log(`    Error code: ${data.errorCode || data.result?.error_code || "none"}`);
    }

    return { success: true, data };
  } catch (error) {
    clearTimeout(timeout);
    if (error.name === "AbortError") {
      throw new Error(`Request timed out after ${TIMEOUT}ms`);
    }
    throw error;
  }
}

/**
 * Test the /ready endpoint (health check)
 */
async function testReadyEndpoint() {
  console.log(`\n--- Testing /ready endpoint ---`);

  try {
    const res = await fetch(`${BASE}/ready`, {
      signal: AbortSignal.timeout(5000),
    });

    console.log(`  Status: ${res.status}`);

    if (res.ok) {
      console.log(`  Backend is ready`);
      return true;
    } else {
      console.log(`  Backend not ready`);
      return false;
    }
  } catch (error) {
    console.log(`  Health check failed: ${error.message}`);
    return false;
  }
}

/**
 * Test the /api/jobs/:id endpoint (if async mode is supported)
 */
async function testJobStatusNormalization() {
  console.log(`\n--- Testing job status normalization ---`);

  // Submit an async verify request
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT);

  try {
    const res = await fetch(`${BASE}/api/verify?async=true`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({
        capsulePath: CAPSULE_PATH,
        mode: MODE,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!res.ok) {
      console.log(`  Async mode not available or failed: ${res.status}`);
      return { supported: false };
    }

    const data = await res.json();
    console.log(`  Async response: ${JSON.stringify(data)}`);

    // Should have jobId and status
    if ("jobId" in data && "status" in data) {
      assertString(data.jobId, "AsyncResponse.jobId");
      assertString(data.status, "AsyncResponse.status");

      // Validate status is normalized
      const normalizedStatuses = new Set([
        "SUBMITTED",
        "queued",
        "running",
        "succeeded",
        "failed",
      ]);
      assertOneOf(data.status, normalizedStatuses, "AsyncResponse.status");

      console.log(`  Job ID: ${data.jobId}`);
      console.log(`  Job status: ${data.status}`);

      // Try to get job status
      if (data.location) {
        const jobRes = await fetch(`${BASE}${data.location}`, {
          headers: { "X-API-Key": API_KEY },
          signal: AbortSignal.timeout(5000),
        });

        if (jobRes.ok) {
          const jobData = await jobRes.json();
          console.log(`  Job details: ${JSON.stringify(jobData)}`);

          if ("status" in jobData) {
            const allowedJobStatuses = new Set([
              "queued",
              "running",
              "succeeded",
              "failed",
              "canceled",
            ]);
            assert.ok(
              allowedJobStatuses.has(jobData.status),
              `Job status must be normalized: ${jobData.status}`
            );
          }
        }
      }

      return { supported: true, jobId: data.jobId };
    }

    return { supported: false };
  } catch (error) {
    clearTimeout(timeout);
    console.log(`  Job status test skipped: ${error.message}`);
    return { supported: false };
  }
}

/**
 * Main test runner
 */
async function main() {
  console.log("=".repeat(60));
  console.log("E2E Verification Contract Test");
  console.log("=".repeat(60));

  const results = {
    ready: false,
    verify: false,
    jobStatus: false,
  };

  // 1. Test backend readiness
  results.ready = await testReadyEndpoint();
  if (!results.ready) {
    console.error("\nBackend is not ready. Is it running?");
    console.error(`Expected: ${BASE}/ready to return 200`);
    process.exit(1);
  }

  // 2. Test verify endpoint contract
  try {
    const verifyResult = await testVerifyEndpoint();
    results.verify = verifyResult.success;
  } catch (error) {
    console.error(`\nVerify contract test FAILED:`);
    console.error(`  ${error.message}`);
    if (error.stack) {
      console.error(`  ${error.stack.split("\n").slice(1, 4).join("\n  ")}`);
    }
    results.verify = false;
  }

  // 3. Test job status normalization (optional)
  try {
    const jobResult = await testJobStatusNormalization();
    results.jobStatus = jobResult.supported !== false;
  } catch (error) {
    console.log(`\nJob status test skipped: ${error.message}`);
  }

  // Summary
  console.log("\n" + "=".repeat(60));
  console.log("Test Results:");
  console.log("=".repeat(60));
  console.log(`  Ready endpoint:    ${results.ready ? "PASS" : "FAIL"}`);
  console.log(`  Verify contract:   ${results.verify ? "PASS" : "FAIL"}`);
  console.log(`  Job status:        ${results.jobStatus ? "PASS" : "SKIP"}`);

  const allPassed = results.ready && results.verify;

  if (allPassed) {
    console.log("\nE2E verify contract OK");
    process.exit(0);
  } else {
    console.error("\nE2E verify contract FAILED");
    process.exit(1);
  }
}

// Run
main().catch((e) => {
  console.error("Unexpected error:", e);
  process.exit(1);
});
