/**
 * Engine module - Backend adapters for the UI.
 *
 * Re-exports the Flask/Relay backend adapter and engine factory.
 */

// Engine factory and configuration
export {
  createEngine,
  getEngine,
  resetEngine,
  getDefaultConfig,
  EngineConfigError,
  resolveConfig,
} from './engine.js'

// Base class
export { ExecutionEngine } from './base.js'

// Flask implementation
export {
  FlaskEngine,
  JobStatus,
  normalizeJobStatus,
  normalizeVerifyReport,
  getLayerById,
  isLayerOk,
  STATUS_CLASSES,
  getStatusClass,
} from './impl/flask.js'
