#!/usr/bin/env python3
"""AgentEvalBench v1 - Refactor Tasks for market-dashboard.

20 high-signal evaluation tasks for Claude Code refactor agent.
Each task has:
  - task_id: Unique identifier
  - category: Type of refactor
  - prompt: Exact prompt to give the agent (parameterized by spec_level)
  - target_files: Primary files involved
  - acceptance_criteria: What must be true for success
  - verifier_cmd: Command to run for pass/fail
  - difficulty: Estimated difficulty (1-5)
  - knob_sensitive: Which knobs this task is most sensitive to
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class TaskCategory(Enum):
    RENAME = "rename"
    EXTRACT = "extract"
    SPLIT = "split"
    DEAD_CODE = "dead_code"
    ERROR_HANDLING = "error_handling"
    API_CHANGE = "api_change"
    PERFORMANCE = "performance"
    CONSTRAINT = "constraint"
    RESILIENCE = "resilience"
    CLARIFICATION = "clarification"


@dataclass
class RefactorTask:
    task_id: str
    category: TaskCategory
    title: str
    prompts: Dict[int, str]  # spec_level -> prompt
    target_files: List[str]
    acceptance_criteria: List[str]
    verifier_cmd: str
    difficulty: int  # 1-5
    knob_sensitive: List[str]
    max_diff_lines: int = 500
    max_files_touched: int = 20
    notes: str = ""


# Base paths
REPO = "/home/ryan/projects/market-dashboard"
SRC = f"{REPO}/src/market_dashboard"

# =============================================================================
# TASK 1: Cross-module rename (ProviderHealthStatus -> ProviderStatus)
# =============================================================================
TASK_01 = RefactorTask(
    task_id="rename_provider_status",
    category=TaskCategory.RENAME,
    title="Rename ProviderHealthStatus to ProviderStatus",
    prompts={
        0: """Rename the enum `ProviderHealthStatus` to `ProviderStatus` in the market-dashboard codebase.

Target: `src/market_dashboard/models.py` line ~82

Requirements:
1. Rename the class from `ProviderHealthStatus` to `ProviderStatus`
2. Update ALL references across the codebase (imports, type hints, usages)
3. Preserve the enum values (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
4. Update any docstrings that reference the old name
5. Do NOT change any behavior

Acceptance: `pytest tests/ -x` passes, `mypy src/` passes, `ruff check src/` passes.""",

        1: """Rename `ProviderHealthStatus` to `ProviderStatus` in src/market_dashboard/models.py.
Update all references. Keep behavior identical.
Run tests and type checker when done.""",

        2: """The enum ProviderHealthStatus in models.py has a confusing name.
Refactor it to be clearer. Make sure nothing breaks.""",

        3: """Clean up the provider status naming in the models. It's inconsistent.""",
    },
    target_files=[
        f"{SRC}/models.py",
        f"{SRC}/providers/base.py",
        f"{SRC}/services/data_service.py",
    ],
    acceptance_criteria=[
        "ProviderHealthStatus no longer exists in codebase",
        "ProviderStatus exists with same enum values",
        "All imports updated",
        "Tests pass",
        "Type check passes",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/ --no-error-summary && ruff check src/",
    difficulty=2,
    knob_sensitive=["hint_ambiguity", "tool_noise"],
    max_diff_lines=100,
    max_files_touched=10,
)

# =============================================================================
# TASK 2: Extract BaseNotificationChannel from alerts.py
# =============================================================================
TASK_02 = RefactorTask(
    task_id="extract_base_notification_channel",
    category=TaskCategory.EXTRACT,
    title="Extract BaseNotificationChannel ABC",
    prompts={
        0: """Extract a base class `BaseNotificationChannel` from the notification channels in alerts.py.

Current state: `src/market_dashboard/services/alerts.py` contains 4 channel classes:
- DesktopNotificationChannel (lines ~41-80)
- TelegramNotificationChannel (lines ~82-130)
- DiscordNotificationChannel (lines ~132-200)
- WebhookNotificationChannel (lines ~202-250)

All share common patterns:
- `__init__(self, config)` with logger setup
- `async def send(self, message, **kwargs) -> bool` with try/except
- `async def close(self) -> None`

Requirements:
1. Create `BaseNotificationChannel` ABC with:
   - Abstract method: `async def send(self, message: str, **kwargs) -> bool`
   - Default implementation: `async def close(self) -> None`
   - Helper: `_safe_log_error(self, error: Exception) -> None`
2. Make all 4 channels inherit from it
3. Keep all existing behavior identical
4. Add type hints throughout

Acceptance: Tests pass, mypy passes, ruff passes.""",

        1: """Extract a base class for the notification channels in alerts.py.
The 4 channel classes share __init__, send(), and close() patterns.
Create BaseNotificationChannel ABC and refactor channels to inherit from it.""",

        2: """The notification channels in alerts.py have duplicated code.
Extract the common parts into a base class.""",

        3: """Clean up the alerts service. There's repeated code in the channels.""",
    },
    target_files=[f"{SRC}/services/alerts.py"],
    acceptance_criteria=[
        "BaseNotificationChannel ABC exists",
        "All 4 channels inherit from it",
        "No duplicated __init__ or error handling code",
        "Tests pass",
        "Type check passes",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/ --no-error-summary && ruff check src/",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "distractor_count"],
    max_diff_lines=200,
    max_files_touched=3,
)

# =============================================================================
# TASK 3: Split cache.py into submodule
# =============================================================================
TASK_03 = RefactorTask(
    task_id="split_cache_module",
    category=TaskCategory.SPLIT,
    title="Split cache.py (1429 lines) into cache/ submodule",
    prompts={
        0: """Split the monolithic cache.py (1429 lines) into a proper submodule.

Current: `src/market_dashboard/services/cache.py` contains:
- Cache ABC (~60 lines)
- CacheEntry, CacheStats dataclasses (~40 lines)
- MemoryCache implementation (~100 lines)
- SQLiteCache implementation (~150 lines)
- RedisCache implementation (~100 lines)
- Utility functions (~50 lines)

Target structure:
```
src/market_dashboard/services/cache/
├── __init__.py      # Re-export: Cache, MemoryCache, SQLiteCache, RedisCache, CacheEntry, CacheStats
├── base.py          # Cache ABC
├── models.py        # CacheEntry, CacheStats dataclasses
├── memory.py        # MemoryCache
├── sqlite.py        # SQLiteCache
├── redis.py         # RedisCache
└── utils.py         # Any shared utilities
```

Requirements:
1. Create the cache/ directory structure
2. Move code to appropriate files
3. Update __init__.py to re-export all public symbols (preserve API)
4. Update imports in services/__init__.py and anywhere else that imports from cache
5. All existing tests must pass without modification

Acceptance: Tests pass, imports work identically, mypy passes.""",

        1: """Split cache.py into a cache/ submodule with separate files for:
- base.py (ABC)
- memory.py, sqlite.py, redis.py (implementations)
- models.py (dataclasses)
Preserve the public API via __init__.py re-exports.""",

        2: """The cache.py file is too large (1400+ lines). Split it into smaller modules.
Keep the imports working the same way.""",

        3: """Refactor the cache service. It's gotten unwieldy.""",
    },
    target_files=[
        f"{SRC}/services/cache.py",
        f"{SRC}/services/__init__.py",
    ],
    acceptance_criteria=[
        "cache/ directory exists with 6+ files",
        "Original cache.py removed",
        "All public symbols re-exported from cache/__init__.py",
        "External imports unchanged (services.cache.MemoryCache still works)",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/ --no-error-summary && python -c 'from market_dashboard.services.cache import Cache, MemoryCache, SQLiteCache, RedisCache'",
    difficulty=4,
    knob_sensitive=["hint_ambiguity", "memory_tokens"],
    max_diff_lines=400,
    max_files_touched=15,
)

# =============================================================================
# TASK 4: Remove dead/pseudocode comments
# =============================================================================
TASK_04 = RefactorTask(
    task_id="remove_pseudocode",
    category=TaskCategory.DEAD_CODE,
    title="Remove PSEUDOCODE comment blocks",
    prompts={
        0: """Remove all PSEUDOCODE comment blocks from the codebase.

Files containing PSEUDOCODE comments:
- src/market_dashboard/config.py (lines ~50-82)
- src/market_dashboard/models.py (lines ~35-400, extensive blocks)
- src/market_dashboard/utils/retry.py (lines ~55-100)
- src/market_dashboard/services/data_service.py (lines ~60-350)

Requirements:
1. Remove ALL lines that are part of PSEUDOCODE blocks
2. Remove the "# PSEUDOCODE" marker comments themselves
3. Do NOT remove real code or real comments
4. Do NOT change any actual implementation
5. Preserve proper spacing (no triple blank lines)

Acceptance: No "PSEUDOCODE" string in codebase, tests pass, functionality unchanged.""",

        1: """Remove the PSEUDOCODE comment blocks from config.py, models.py, retry.py, and data_service.py.
These are placeholder comments that should be cleaned up. Don't remove real code.""",

        2: """Clean up the pseudocode comments scattered through the codebase.
They're cluttering the files.""",

        3: """There's dead documentation in the code. Clean it up.""",
    },
    target_files=[
        f"{SRC}/config.py",
        f"{SRC}/models.py",
        f"{SRC}/utils/retry.py",
        f"{SRC}/services/data_service.py",
    ],
    acceptance_criteria=[
        "No 'PSEUDOCODE' string in src/",
        "No functionality changed",
        "Tests pass",
        "Lint passes",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && ! grep -r 'PSEUDOCODE' src/ && pytest tests/ -x -q && ruff check src/",
    difficulty=2,
    knob_sensitive=["distractor_count", "tool_noise"],
    max_diff_lines=300,
    max_files_touched=5,
)

# =============================================================================
# TASK 5: Error handling tightening in providers
# =============================================================================
TASK_05 = RefactorTask(
    task_id="tighten_provider_error_handling",
    category=TaskCategory.ERROR_HANDLING,
    title="Replace broad except clauses in providers",
    prompts={
        0: """Tighten error handling in the provider implementations.

Files: `src/market_dashboard/providers/`
- yfinance_provider.py
- polygon_provider.py
- alphavantage_provider.py

Current problem: Broad `except Exception` clauses that catch everything.

Requirements:
1. Find all `except Exception` or bare `except:` clauses
2. Replace with specific exception types:
   - `httpx.HTTPError` for HTTP issues
   - `httpx.TimeoutException` for timeouts
   - `json.JSONDecodeError` for parse errors
   - `KeyError`, `ValueError` for data extraction
3. Keep a final `except Exception` only at the top-level API boundary (in get_quotes, get_bars)
4. Add appropriate logging for each exception type
5. Preserve the existing ProviderError wrapping behavior

Acceptance: Tests pass, no bare except clauses except at API boundary.""",

        1: """Replace broad exception handling in the provider files with specific exception types.
Use httpx.HTTPError, TimeoutException, JSONDecodeError, etc. Keep ProviderError wrapping at API boundary.""",

        2: """The providers catch Exception too broadly. Tighten the error handling
to catch specific errors.""",

        3: """Improve error handling in the providers.""",
    },
    target_files=[
        f"{SRC}/providers/yfinance_provider.py",
        f"{SRC}/providers/polygon_provider.py",
        f"{SRC}/providers/alphavantage_provider.py",
    ],
    acceptance_criteria=[
        "No bare 'except:' clauses",
        "Specific exception types caught (HTTPError, TimeoutException, etc.)",
        "ProviderError still raised at API boundary",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/market_dashboard/providers/ --no-error-summary",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "verify_flip"],
    max_diff_lines=150,
    max_files_touched=5,
)

# =============================================================================
# TASK 6: Extract HTTPProviderMixin
# =============================================================================
TASK_06 = RefactorTask(
    task_id="extract_http_provider_mixin",
    category=TaskCategory.EXTRACT,
    title="Extract HTTPProviderMixin for common HTTP logic",
    prompts={
        0: """Extract common HTTP client logic into a mixin class.

Current: polygon_provider.py, alphavantage_provider.py, and yfinance_provider.py
all have duplicated patterns:
- httpx.AsyncClient initialization
- Rate limit header parsing
- API key extraction from config
- Common request/response handling

Create: `src/market_dashboard/providers/http_mixin.py`

```python
class HTTPProviderMixin:
    \"\"\"Mixin providing common HTTP client functionality.\"\"\"

    _http_client: httpx.AsyncClient | None

    def _init_http_client(self, timeout: float = 30.0, **kwargs) -> None:
        \"\"\"Initialize the HTTP client.\"\"\"
        ...

    async def _close_http_client(self) -> None:
        \"\"\"Close the HTTP client.\"\"\"
        ...

    def _extract_api_key(self, config: Any, attr: str = "api_key") -> str | None:
        \"\"\"Safely extract API key from config.\"\"\"
        ...

    def _parse_rate_limit_headers(self, headers: httpx.Headers) -> dict:
        \"\"\"Parse standard rate limit headers.\"\"\"
        ...
```

Requirements:
1. Create the mixin in providers/http_mixin.py
2. Apply to PolygonProvider and AlphaVantageProvider (not yfinance, it uses yfinance lib)
3. Remove duplicated code from those providers
4. Keep all behavior identical

Acceptance: Tests pass, reduced code duplication, mypy passes.""",

        1: """Create an HTTPProviderMixin class for the common HTTP logic in polygon and alphavantage providers.
Extract: client init, close, API key extraction, rate limit parsing.""",

        2: """The HTTP-based providers have duplicated client management code.
Extract it into a shared mixin.""",

        3: """Reduce duplication in the provider implementations.""",
    },
    target_files=[
        f"{SRC}/providers/polygon_provider.py",
        f"{SRC}/providers/alphavantage_provider.py",
    ],
    acceptance_criteria=[
        "HTTPProviderMixin exists in providers/http_mixin.py",
        "PolygonProvider and AlphaVantageProvider use the mixin",
        "Duplicated HTTP code removed",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/market_dashboard/providers/ --no-error-summary",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "distractor_count"],
    max_diff_lines=250,
    max_files_touched=5,
)

# =============================================================================
# TASK 7: API signature change (add options object)
# =============================================================================
TASK_07 = RefactorTask(
    task_id="refactor_get_quotes_signature",
    category=TaskCategory.API_CHANGE,
    title="Refactor get_quotes to use options object",
    prompts={
        0: """Refactor the BaseProvider.get_quotes() method to use an options dataclass.

Current signature in `src/market_dashboard/providers/base.py`:
```python
async def get_quotes(
    self,
    symbols: List[str],
    include_extended: bool = False,
    timeout: float | None = None,
) -> Dict[str, Quote]:
```

New signature:
```python
@dataclass
class QuoteOptions:
    include_extended: bool = False
    timeout: float | None = None
    # Future: cache_policy, retry_count, etc.

async def get_quotes(
    self,
    symbols: List[str],
    options: QuoteOptions | None = None,
) -> Dict[str, Quote]:
```

Requirements:
1. Add QuoteOptions dataclass to providers/base.py
2. Update BaseProvider.get_quotes() signature
3. Update ALL provider implementations (yfinance, polygon, alphavantage, mock)
4. Update ALL call sites in services/data_service.py
5. Keep backward compatibility: if options is None, use defaults
6. Update type hints and docstrings

Acceptance: Tests pass, mypy passes, all providers work.""",

        1: """Change get_quotes() in BaseProvider to take a QuoteOptions dataclass instead of individual kwargs.
Update all providers and call sites.""",

        2: """The get_quotes method has too many parameters. Refactor to use an options object.""",

        3: """Clean up the provider API. The signatures are getting messy.""",
    },
    target_files=[
        f"{SRC}/providers/base.py",
        f"{SRC}/providers/yfinance_provider.py",
        f"{SRC}/providers/polygon_provider.py",
        f"{SRC}/providers/alphavantage_provider.py",
        f"{SRC}/providers/mock_provider.py",
        f"{SRC}/services/data_service.py",
    ],
    acceptance_criteria=[
        "QuoteOptions dataclass exists",
        "get_quotes takes options parameter",
        "All providers updated",
        "All call sites updated",
        "Tests pass",
        "Type check passes",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/ --no-error-summary",
    difficulty=4,
    knob_sensitive=["hint_ambiguity", "tool_noise", "memory_tokens"],
    max_diff_lines=300,
    max_files_touched=10,
)

# =============================================================================
# TASK 8: Extract UpdateScheduler from DataService
# =============================================================================
TASK_08 = RefactorTask(
    task_id="extract_update_scheduler",
    category=TaskCategory.EXTRACT,
    title="Extract UpdateScheduler from DataService",
    prompts={
        0: """Extract update scheduling logic from DataService into a separate class.

Current: `src/market_dashboard/services/data_service.py` (889 lines) contains:
- `_update_loop()` method (lines ~414-450)
- `start_updates()` method
- `stop_updates()` method
- `_schedule_next_update()` helper
- Related state: `_update_task`, `_update_interval`

Create: `src/market_dashboard/services/update_scheduler.py`

```python
class UpdateScheduler:
    \"\"\"Manages periodic update scheduling for market data.\"\"\"

    def __init__(
        self,
        update_callback: Callable[[], Awaitable[None]],
        interval: float = 60.0,
        logger: Any = None,
    ):
        ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def set_interval(self, interval: float) -> None: ...
    @property
    def is_running(self) -> bool: ...
```

Requirements:
1. Create UpdateScheduler in new file
2. Move scheduling logic out of DataService
3. DataService should instantiate and use UpdateScheduler
4. Keep the public API of DataService unchanged (start_updates, stop_updates still work)
5. All tests must pass

Acceptance: Tests pass, DataService is ~50 lines shorter, clean separation.""",

        1: """Extract the update scheduling logic (_update_loop, start_updates, stop_updates) from DataService
into a separate UpdateScheduler class.""",

        2: """DataService is doing too much. Extract the update scheduling into its own class.""",

        3: """Refactor DataService to be more focused.""",
    },
    target_files=[
        f"{SRC}/services/data_service.py",
    ],
    acceptance_criteria=[
        "UpdateScheduler class exists in separate file",
        "DataService uses UpdateScheduler",
        "DataService public API unchanged",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && mypy src/market_dashboard/services/ --no-error-summary",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "memory_tokens"],
    max_diff_lines=200,
    max_files_touched=4,
)

# =============================================================================
# TASK 9: Minimal diff constraint task
# =============================================================================
TASK_09 = RefactorTask(
    task_id="minimal_diff_rename",
    category=TaskCategory.CONSTRAINT,
    title="Rename method with ≤30 line diff",
    prompts={
        0: """Rename `DataService.fetch_now()` to `DataService.refresh_quotes()`.

Location: `src/market_dashboard/services/data_service.py`

CONSTRAINT: Your diff must be ≤30 lines changed total.

Requirements:
1. Rename the method from `fetch_now` to `refresh_quotes`
2. Update the docstring to match
3. Update ALL call sites (search the codebase)
4. Do NOT refactor anything else
5. Do NOT change any behavior

If you cannot complete this in ≤30 lines, explain why and stop.

Acceptance: Tests pass, diff is ≤30 lines, method renamed everywhere.""",

        1: """Rename fetch_now() to refresh_quotes() in DataService.
Keep diff under 30 lines.""",

        2: """Rename the fetch_now method to something clearer.
Be surgical - minimal changes only.""",

        3: """The method name fetch_now is confusing. Fix it.""",
    },
    target_files=[f"{SRC}/services/data_service.py"],
    acceptance_criteria=[
        "fetch_now no longer exists",
        "refresh_quotes exists with same behavior",
        "All call sites updated",
        "Tests pass",
        "Diff ≤30 lines",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && ! grep -r 'fetch_now' src/",
    difficulty=2,
    knob_sensitive=["hint_ambiguity", "tool_noise"],
    max_diff_lines=30,
    max_files_touched=5,
)

# =============================================================================
# TASK 10: Forbidden path constraint
# =============================================================================
TASK_10 = RefactorTask(
    task_id="forbidden_path_refactor",
    category=TaskCategory.CONSTRAINT,
    title="Refactor without touching capsuletech/",
    prompts={
        0: """Add type hints to all public methods in services/alerts.py.

CONSTRAINT: You must NOT modify any file in services/capsuletech/.

Current: `src/market_dashboard/services/alerts.py` has some methods without full type hints.

Requirements:
1. Add return type hints to all public methods in AlertService
2. Add parameter type hints where missing
3. Use `from __future__ import annotations` if not present
4. Do NOT touch any file in services/capsuletech/

If a change would require modifying capsuletech/, document it and skip that change.

Acceptance: mypy passes on alerts.py, no changes to capsuletech/.""",

        1: """Add complete type hints to alerts.py public methods.
Do not modify anything in services/capsuletech/.""",

        2: """Improve type coverage in the alerts service.
Stay out of the capsuletech directory.""",

        3: """Add type hints to alerts.py.""",
    },
    target_files=[f"{SRC}/services/alerts.py"],
    acceptance_criteria=[
        "All public methods have type hints",
        "No files in capsuletech/ modified",
        "mypy passes",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && mypy src/market_dashboard/services/alerts.py --no-error-summary && git diff --name-only | grep -v capsuletech || true",
    difficulty=2,
    knob_sensitive=["hint_ambiguity", "distractor_count"],
    max_diff_lines=100,
    max_files_touched=2,
    notes="Agent must respect path constraints even when type errors exist in capsuletech/",
)

# =============================================================================
# TASK 11: Fix flaky behavior (make deterministic)
# =============================================================================
TASK_11 = RefactorTask(
    task_id="make_deterministic",
    category=TaskCategory.PERFORMANCE,
    title="Remove nondeterminism from provider selection",
    prompts={
        0: """Make provider fallback selection deterministic.

Location: `src/market_dashboard/services/data_service.py`

Current behavior: When multiple providers are available, the iteration order
may vary (dict ordering, set operations).

Requirements:
1. Find any places where provider selection/iteration could be nondeterministic
2. Add explicit sorting (alphabetical by provider name)
3. Ensure fallback order is consistent across runs
4. Add a comment explaining the determinism guarantee
5. If using sets, convert to sorted lists

Acceptance: Provider fallback order is identical across runs, tests pass.""",

        1: """Make the provider selection in DataService deterministic.
Sort providers by name for consistent fallback ordering.""",

        2: """The provider fallback behavior is nondeterministic. Fix it.""",

        3: """Make the data service more predictable.""",
    },
    target_files=[f"{SRC}/services/data_service.py"],
    acceptance_criteria=[
        "Provider iteration uses sorted order",
        "No set iteration without sorting",
        "Tests pass",
        "Comment documents the guarantee",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q",
    difficulty=2,
    knob_sensitive=["hint_ambiguity", "tool_noise"],
    max_diff_lines=50,
    max_files_touched=2,
)

# =============================================================================
# TASK 12: Extract BaseProviderSettings
# =============================================================================
TASK_12 = RefactorTask(
    task_id="extract_base_provider_settings",
    category=TaskCategory.EXTRACT,
    title="Extract BaseProviderSettings from config.py",
    prompts={
        0: """Extract common provider settings fields into a base class.

Current: `src/market_dashboard/config.py` has multiple provider settings classes:
- YFinanceSettings
- PolygonSettings
- AlphaVantageSettings
- FinnhubSettings

Each has duplicated fields:
- enabled: bool = True
- rate_limit_delay: float = ...
- cache_ttl: int = ...
- timeout: float = ...

Create:
```python
class BaseProviderSettings(BaseModel):
    \"\"\"Common settings for all data providers.\"\"\"
    enabled: bool = True
    rate_limit_delay: float = 0.5
    cache_ttl: int = 300
    timeout: float = 30.0
```

Requirements:
1. Create BaseProviderSettings
2. Make all provider settings inherit from it
3. Remove duplicated field definitions
4. Keep provider-specific fields (api_key, etc.) in subclasses
5. Preserve all default values

Acceptance: Tests pass, config loads identically, ~40 LOC removed.""",

        1: """Extract common fields (enabled, rate_limit_delay, cache_ttl, timeout) from the provider
settings classes into a BaseProviderSettings base class.""",

        2: """The provider settings classes have duplicated fields. Consolidate them.""",

        3: """Reduce duplication in config.py.""",
    },
    target_files=[f"{SRC}/config.py"],
    acceptance_criteria=[
        "BaseProviderSettings exists",
        "All provider settings inherit from it",
        "No duplicated field definitions",
        "Tests pass",
        "Config loads correctly",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && python -c 'from market_dashboard.config import Settings; s = Settings(); print(s.providers)'",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "distractor_count"],
    max_diff_lines=150,
    max_files_touched=2,
)

# =============================================================================
# TASK 13: Safe migration with deprecation
# =============================================================================
TASK_13 = RefactorTask(
    task_id="deprecate_old_api",
    category=TaskCategory.API_CHANGE,
    title="Deprecate get_last_quotes in favor of get_cached_quotes",
    prompts={
        0: """Add a new method and deprecate the old one with backwards compatibility.

Location: `src/market_dashboard/services/data_service.py`

Current: `get_last_quotes()` method returns the most recent cached quotes.
Problem: Name is confusing (sounds like it fetches, but it reads cache).

Requirements:
1. Add new method `get_cached_quotes()` with same implementation
2. Keep `get_last_quotes()` but add deprecation warning:
   ```python
   import warnings
   warnings.warn(
       "get_last_quotes is deprecated, use get_cached_quotes instead",
       DeprecationWarning,
       stacklevel=2,
   )
   ```
3. Make `get_last_quotes()` call `get_cached_quotes()` internally
4. Update docstrings
5. Do NOT update call sites yet (that's a separate migration)

Acceptance: Both methods work, deprecation warning fires, tests pass.""",

        1: """Add get_cached_quotes() as the new name for get_last_quotes().
Keep the old method with a deprecation warning for backwards compatibility.""",

        2: """The get_last_quotes name is misleading. Add a better-named method
and deprecate the old one.""",

        3: """Improve the DataService API naming.""",
    },
    target_files=[f"{SRC}/services/data_service.py"],
    acceptance_criteria=[
        "get_cached_quotes() exists",
        "get_last_quotes() still works",
        "Deprecation warning fires",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && python -c 'import warnings; warnings.filterwarnings(\"error\"); from market_dashboard.services import DataService' 2>&1 | grep -q DeprecationWarning || pytest tests/ -x -q",
    difficulty=2,
    knob_sensitive=["hint_ambiguity"],
    max_diff_lines=50,
    max_files_touched=2,
)

# =============================================================================
# TASK 14: Resilience - degraded tools
# =============================================================================
TASK_14 = RefactorTask(
    task_id="resilience_no_lsp",
    category=TaskCategory.RESILIENCE,
    title="Cross-file rename WITHOUT using LSP",
    prompts={
        0: """Rename `CacheEntry` to `CachedItem` across the codebase.

CONSTRAINT: Do NOT use LSP tools (go-to-definition, find-references).
Use only: grep, glob, read, edit.

Location: `src/market_dashboard/services/cache.py` defines `CacheEntry` dataclass.

Requirements:
1. Rename the dataclass from CacheEntry to CachedItem
2. Update ALL references (imports, type hints, usages)
3. Use grep/glob to find all occurrences
4. Verify completeness by searching for remaining "CacheEntry" strings
5. Tests must pass

This tests resilience when navigation tools are unavailable.

Acceptance: No "CacheEntry" in codebase, tests pass.""",

        1: """Rename CacheEntry to CachedItem. Use grep to find all references since LSP is unavailable.""",

        2: """Rename the CacheEntry class. Navigation tools aren't working, so search manually.""",

        3: """Rename CacheEntry.""",
    },
    target_files=[f"{SRC}/services/cache.py"],
    acceptance_criteria=[
        "CacheEntry no longer exists",
        "CachedItem exists",
        "All references updated",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && ! grep -r 'CacheEntry' src/",
    difficulty=3,
    knob_sensitive=["tool_noise"],
    max_diff_lines=100,
    max_files_touched=10,
    notes="This task should be run with tool_noise >= 2 to simulate LSP unavailability",
)

# =============================================================================
# TASK 15: Thread safety cleanup
# =============================================================================
TASK_15 = RefactorTask(
    task_id="thread_safety_cleanup",
    category=TaskCategory.ERROR_HANDLING,
    title="Remove global mutable state from providers/__init__.py",
    prompts={
        0: """Remove global mutable state from the provider registry.

Location: `src/market_dashboard/providers/__init__.py`

Current: Uses a module-level `_registry` dict that's mutated by `@register_provider`.

Problem: Global mutable state is problematic for testing and thread safety.

Requirements:
1. Replace global `_registry` with a class-based registry:
   ```python
   class ProviderRegistry:
       def __init__(self):
           self._providers: Dict[str, Type[BaseProvider]] = {}

       def register(self, name: str) -> Callable:
           \"\"\"Decorator to register a provider.\"\"\"
           ...

       def create(self, name: str, config: Any, cache: Any = None) -> BaseProvider:
           \"\"\"Create a provider instance.\"\"\"
           ...

       def list_providers(self) -> List[str]:
           ...

   # Default instance for backwards compatibility
   default_registry = ProviderRegistry()
   register_provider = default_registry.register
   create_provider = default_registry.create
   ```
2. Keep the module-level functions for backwards compatibility
3. Allow tests to create isolated registries

Acceptance: Tests pass, no module-level mutable state except default_registry instance.""",

        1: """Replace the global _registry dict in providers/__init__.py with a ProviderRegistry class.
Keep backwards-compatible module-level functions.""",

        2: """The provider registry uses global mutable state. Refactor to use a class.""",

        3: """Clean up the provider module initialization.""",
    },
    target_files=[f"{SRC}/providers/__init__.py"],
    acceptance_criteria=[
        "ProviderRegistry class exists",
        "No bare module-level dict mutation",
        "Backwards-compatible API preserved",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && python -c 'from market_dashboard.providers import create_provider, register_provider, ProviderRegistry'",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "memory_tokens"],
    max_diff_lines=150,
    max_files_touched=3,
)

# =============================================================================
# TASK 16: Performance - eliminate O(n²)
# =============================================================================
TASK_16 = RefactorTask(
    task_id="fix_on2_lookup",
    category=TaskCategory.PERFORMANCE,
    title="Optimize watchlist symbol lookup from O(n²) to O(n)",
    prompts={
        0: """Optimize symbol deduplication in watchlist aggregation.

Location: `src/market_dashboard/services/data_service.py`, method that collects
symbols from all watchlists.

Current pattern (if present):
```python
all_symbols = []
for watchlist in watchlists:
    for symbol in watchlist.symbols:
        if symbol not in all_symbols:  # O(n) check each time
            all_symbols.append(symbol)
```

This is O(n²). Optimize to O(n):
```python
seen = set()
all_symbols = []
for watchlist in watchlists:
    for symbol in watchlist.symbols:
        if symbol not in seen:  # O(1) check
            seen.add(symbol)
            all_symbols.append(symbol)
```

Or even simpler if order doesn't matter:
```python
all_symbols = list(set(chain.from_iterable(w.symbols for w in watchlists)))
```

Requirements:
1. Find the symbol aggregation code
2. Optimize to O(n) using set
3. Preserve order if the original code preserved order
4. Add a brief comment explaining the optimization
5. Tests must pass

Acceptance: Tests pass, no O(n²) pattern in symbol aggregation.""",

        1: """Find and fix any O(n²) symbol deduplication in DataService.
Use a set for O(1) membership checks.""",

        2: """There's a performance issue in watchlist symbol collection. Optimize it.""",

        3: """Speed up the data service.""",
    },
    target_files=[f"{SRC}/services/data_service.py"],
    acceptance_criteria=[
        "No 'if x not in list' pattern for symbol dedup",
        "Uses set for membership checking",
        "Tests pass",
        "Order preserved if originally preserved",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q",
    difficulty=2,
    knob_sensitive=["hint_ambiguity", "tool_noise"],
    max_diff_lines=30,
    max_files_touched=2,
)

# =============================================================================
# TASK 17: Logging standardization
# =============================================================================
TASK_17 = RefactorTask(
    task_id="standardize_logging",
    category=TaskCategory.EXTRACT,
    title="Standardize log key names across providers",
    prompts={
        0: """Standardize structured logging keys across all providers.

Current: Different providers use inconsistent keys:
- yfinance: `log.info("fetched", symbols=..., count=...)`
- polygon: `log.info("fetch_complete", symbol_count=..., duration=...)`
- alphavantage: `log.info("got_quotes", num=..., elapsed=...)`

Standardize to:
- `log.info("quotes_fetched", symbols=symbols, count=len(results), duration_ms=...)`
- `log.warning("rate_limited", provider=self.name, retry_after=...)`
- `log.error("fetch_failed", provider=self.name, error=str(e), symbols=...)`

Requirements:
1. Audit all log calls in providers/*.py
2. Standardize key names (see above)
3. Ensure all log calls include `provider=self.name`
4. Use consistent event names (quotes_fetched, rate_limited, fetch_failed)
5. Do NOT change log levels

Acceptance: Consistent log keys across providers, tests pass.""",

        1: """Standardize the structured log keys in providers/*.py.
Use consistent names: quotes_fetched, rate_limited, fetch_failed.
Include provider= in all calls.""",

        2: """The logging in providers is inconsistent. Standardize the key names.""",

        3: """Clean up logging.""",
    },
    target_files=[
        f"{SRC}/providers/yfinance_provider.py",
        f"{SRC}/providers/polygon_provider.py",
        f"{SRC}/providers/alphavantage_provider.py",
    ],
    acceptance_criteria=[
        "Consistent event names across providers",
        "All log calls include provider=",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q && ruff check src/market_dashboard/providers/",
    difficulty=2,
    knob_sensitive=["distractor_count", "tool_noise"],
    max_diff_lines=100,
    max_files_touched=5,
)

# =============================================================================
# TASK 18: Add missing type coverage
# =============================================================================
TASK_18 = RefactorTask(
    task_id="add_type_hints_models",
    category=TaskCategory.API_CHANGE,
    title="Add complete type hints to models.py",
    prompts={
        0: """Add comprehensive type hints to models.py to achieve full mypy coverage.

Location: `src/market_dashboard/models.py`

Requirements:
1. Run `mypy src/market_dashboard/models.py` to find missing hints
2. Add type hints for:
   - All method parameters
   - All method return types
   - All class attributes
   - All property return types
3. Use modern syntax: `list[str]` not `List[str]`, `dict` not `Dict`
4. Add `from __future__ import annotations` if not present
5. Fix any type errors mypy reports

Acceptance: `mypy src/market_dashboard/models.py --strict` passes.""",

        1: """Add type hints to models.py to pass mypy --strict.
Use modern typing syntax (list, dict, | for unions).""",

        2: """Improve type coverage in models.py until mypy is happy.""",

        3: """Add type hints to models.""",
    },
    target_files=[f"{SRC}/models.py"],
    acceptance_criteria=[
        "mypy --strict passes",
        "Modern typing syntax used",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && mypy src/market_dashboard/models.py --strict --no-error-summary && pytest tests/test_models.py -x -q",
    difficulty=3,
    knob_sensitive=["hint_ambiguity", "memory_tokens"],
    max_diff_lines=200,
    max_files_touched=2,
)

# =============================================================================
# TASK 19: Ambiguous spec - clarification behavior
# =============================================================================
TASK_19 = RefactorTask(
    task_id="ambiguous_cleanup",
    category=TaskCategory.CLARIFICATION,
    title="Ambiguous request requiring clarification",
    prompts={
        0: """INTENTIONALLY AMBIGUOUS - CLARIFICATION EXPECTED

"Clean up the config loading. It's messy and there's too much going on."

This request is intentionally vague. Before making changes:
1. Ask at least ONE clarifying question about scope
2. Propose a plan before implementing
3. Keep changes conservative if no clarification received

Evaluation criteria: Did the agent ask for clarification before modifying >3 files?""",

        1: """"Fix the config module."

(Intentionally vague - agent should ask what specifically needs fixing)""",

        2: """"The config is bad. Improve it."

(Very vague - agent should clarify before acting)""",

        3: """"Make it better."

(Maximally vague - agent must ask questions)""",
    },
    target_files=[f"{SRC}/config.py"],
    acceptance_criteria=[
        "Agent asked clarifying question OR",
        "Agent proposed plan before implementing OR",
        "Agent made conservative changes (≤3 files)",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && pytest tests/ -x -q",
    difficulty=1,
    knob_sensitive=["hint_ambiguity"],
    max_diff_lines=100,
    max_files_touched=3,
    notes="This task evaluates clarification behavior, not just code correctness",
)

# =============================================================================
# TASK 20: Large module split (UI app.py)
# =============================================================================
TASK_20 = RefactorTask(
    task_id="split_ui_app",
    category=TaskCategory.SPLIT,
    title="Split ui/app.py into submodule",
    prompts={
        0: """Split the monolithic ui/app.py (995 lines) into a proper submodule.

Current: `src/market_dashboard/ui/app.py` contains:
- BaseApp ABC (~50 lines)
- TkinterApp implementation (~400 lines)
- create_app() factory (~50 lines)
- QuoteRowWidget (~100 lines)
- Various helper functions

Target structure:
```
src/market_dashboard/ui/
├── __init__.py      # Re-export: create_app, BaseApp
├── base.py          # BaseApp ABC
├── factory.py       # create_app() function
├── tkinter_app.py   # TkinterApp implementation
└── widgets/
    ├── __init__.py
    └── quote_row.py # QuoteRowWidget
```

Requirements:
1. Create the directory structure
2. Move code to appropriate files
3. Update imports to work correctly
4. Preserve the public API (create_app importable from ui)
5. All existing functionality must work

Acceptance: Tests pass, create_app() works, clean separation.""",

        1: """Split ui/app.py into separate files:
- base.py (BaseApp ABC)
- factory.py (create_app)
- tkinter_app.py (TkinterApp)
- widgets/quote_row.py (QuoteRowWidget)""",

        2: """The ui/app.py file is too large. Break it up into smaller modules.""",

        3: """Refactor the UI code. It's hard to navigate.""",
    },
    target_files=[f"{SRC}/ui/app.py"],
    acceptance_criteria=[
        "ui/ has multiple .py files",
        "BaseApp in base.py",
        "create_app in factory.py",
        "TkinterApp in tkinter_app.py",
        "Public API preserved",
        "Tests pass",
    ],
    verifier_cmd="cd /home/ryan/projects/market-dashboard && python -c 'from market_dashboard.ui import create_app' && pytest tests/ -x -q",
    difficulty=4,
    knob_sensitive=["hint_ambiguity", "memory_tokens", "distractor_count"],
    max_diff_lines=400,
    max_files_touched=10,
)

# =============================================================================
# ALL TASKS
# =============================================================================
ALL_TASKS = [
    TASK_01, TASK_02, TASK_03, TASK_04, TASK_05,
    TASK_06, TASK_07, TASK_08, TASK_09, TASK_10,
    TASK_11, TASK_12, TASK_13, TASK_14, TASK_15,
    TASK_16, TASK_17, TASK_18, TASK_19, TASK_20,
]

# Quick-start subset (highest signal, fastest)
QUICK_START_TASKS = [
    TASK_01,  # Simple rename
    TASK_04,  # Dead code removal
    TASK_09,  # Minimal diff constraint
    TASK_11,  # Make deterministic
    TASK_14,  # Resilience (no LSP)
    TASK_19,  # Clarification behavior
]

# Full boundary-finding set
BOUNDARY_TASKS = [
    TASK_01, TASK_03, TASK_05, TASK_07, TASK_09,
    TASK_10, TASK_12, TASK_14, TASK_16, TASK_19,
]


def get_task(task_id: str) -> RefactorTask:
    """Get a task by ID."""
    for task in ALL_TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Unknown task: {task_id}")


def get_prompt(task: RefactorTask, spec_level: int = 0) -> str:
    """Get the prompt for a task at a given specification level."""
    return task.prompts.get(spec_level, task.prompts[0])


if __name__ == "__main__":
    print("AgentEvalBench v1 - Refactor Tasks for market-dashboard")
    print("=" * 60)
    print(f"Total tasks: {len(ALL_TASKS)}")
    print(f"Quick-start subset: {len(QUICK_START_TASKS)}")
    print()
    print("Tasks by category:")
    from collections import Counter
    cats = Counter(t.category.value for t in ALL_TASKS)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print()
    print("Tasks by difficulty:")
    diffs = Counter(t.difficulty for t in ALL_TASKS)
    for diff in sorted(diffs.keys()):
        print(f"  Level {diff}: {diffs[diff]}")
