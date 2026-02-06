function withCors(resp) {
  const headers = new Headers(resp.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type,Authorization");
  return new Response(resp.body, { status: resp.status, headers });
}

function jsonResponse(data, status = 200) {
  return withCors(
    new Response(JSON.stringify(data), {
      status,
      headers: { "content-type": "application/json" },
    }),
  );
}

function notFound() {
  return withCors(new Response("Not found", { status: 404 }));
}

function normalize(payload, key) {
  if (payload && typeof payload === "object" && !Array.isArray(payload) && payload[key] !== undefined) {
    return payload;
  }
  return { [key]: payload };
}

async function proxyJson(path, env, wrapKey, init = {}) {
  if (!env.RELAY_BASE) {
    return jsonResponse({
      error: "Relay base URL missing",
      hint: "Set RELAY_BASE in wrangler.toml or Worker environment",
    }, 500);
  }
  const target = new URL(path, env.RELAY_BASE);
  const headers = new Headers(init.headers || {});
  headers.set("accept", "application/json");
  const upstream = await fetch(target.toString(), { ...init, headers });
  const text = await upstream.text();
  let payload;
  try {
    payload = JSON.parse(text || "{}");
  } catch (err) {
    return jsonResponse({ error: "Relay returned non-JSON payload" }, 502);
  }
  if (wrapKey) {
    payload = normalize(payload, wrapKey);
  }
  return jsonResponse(payload, upstream.status);
}

async function proxyBinary(path, env) {
  if (!env.RELAY_BASE) {
    return jsonResponse({
      error: "Relay base URL missing",
      hint: "Set RELAY_BASE in wrangler.toml or Worker environment",
    }, 500);
  }
  const target = new URL(path, env.RELAY_BASE);
  const upstream = await fetch(target.toString());
  const headers = new Headers(upstream.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "GET,OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type,Authorization");
  return new Response(upstream.body, { status: upstream.status, headers });
}

async function fetchArtifactMeta(runId, filename, env) {
  if (!env.RELAY_BASE) {
    throw new Error("Relay base URL missing");
  }
  const target = new URL(`/runs/${encodeURIComponent(runId)}/artifact_meta/${encodeURIComponent(filename)}`, env.RELAY_BASE);
  const resp = await fetch(target.toString(), { headers: { accept: "application/json" } });
  if (resp.status !== 200) {
    return null;
  }
  return resp.json();
}

async function handleApi(request, env) {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return jsonResponse({ ok: true, worker: "capsuletech" });
  }
  if (url.pathname === "/api/runs") {
    return proxyJson(`/runs${url.search}`, env, "runs");
  }

  const verifyMatch = url.pathname.match(/^\/api\/runs\/(.+?)\/verify$/);
  if (verifyMatch && request.method === "POST") {
    const runId = decodeURIComponent(verifyMatch[1]);
    return proxyJson(`/runs/${encodeURIComponent(runId)}/verify`, env, null, { method: "POST" });
  }

  const artifactFileMatch = url.pathname.match(/^\/api\/runs\/(.+?)\/artifacts\/(.+)$/);
  if (artifactFileMatch) {
    const runId = decodeURIComponent(artifactFileMatch[1]);
    const filename = decodeURIComponent(artifactFileMatch[2]);
    try {
      const meta = await fetchArtifactMeta(runId, filename, env);
      if (meta && meta.storage === "r2") {
        if (!env.ARTIFACTS_BUCKET) {
          return jsonResponse({ error: "R2 bucket not configured" }, 500);
        }
        const signedUrl = await env.ARTIFACTS_BUCKET.get(meta.object_key, {
          presign: {
            expiration: 300,
          },
        });
        const headers = new Headers({ Location: signedUrl });
        headers.set("Access-Control-Allow-Origin", "*");
        return new Response(null, { status: 302, headers });
      }
    } catch (err) {
      return jsonResponse({ error: err.message }, 500);
    }
    return proxyBinary(`/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(filename)}`, env);
  }

  const artifactListMatch = url.pathname.match(/^\/api\/runs\/(.+?)\/artifacts$/);
  if (artifactListMatch) {
    const runId = decodeURIComponent(artifactListMatch[1]);
    return proxyJson(`/runs/${encodeURIComponent(runId)}/artifacts`, env, "artifacts");
  }

  const detailMatch = url.pathname.match(/^\/api\/runs\/(.+?)$/);
  if (detailMatch && !url.pathname.endsWith("/events")) {
    const runId = decodeURIComponent(detailMatch[1]);
    return proxyJson(`/runs/${encodeURIComponent(runId)}${url.search}`, env, "run");
  }

  const match = url.pathname.match(/^\/api\/runs\/(.+?)\/events$/);
  if (match) {
    const runId = decodeURIComponent(match[1]);
    const qs = url.search ? url.search : "";
    return proxyJson(`/runs/${encodeURIComponent(runId)}/events${qs}`, env, "events");
  }
  return notFound();
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return withCors(new Response(null, { status: 204 }));
    }
    const url = new URL(request.url);
    if (url.pathname.startsWith("/api/")) {
      return handleApi(request, env);
    }
    return withCors(
      new Response(
        JSON.stringify({
          message: "CapsuleTech worker online",
          routes: ["/api/health", "/api/runs", "/api/runs/:id/events"],
        }),
        { headers: { "content-type": "application/json" } },
      ),
    );
  },
};
