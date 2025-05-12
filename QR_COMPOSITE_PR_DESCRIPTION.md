# Add QR Decomposition with Composite Operations for Custom Backends

## Summary

This PR adds support for QR decomposition using `stablehlo.composite` operations for custom JAX backends. The implementation provides two operational modes:
1. For standard platforms (CPU/GPU/ROCm): Uses optimized custom calls for best performance
2. For custom backends: Uses composite operations with appropriate attributes enabling custom implementations

## Key Features

- **Platform-aware implementation**: Automatically detects platform type and applies the appropriate implementation strategy
- **Backward compatible**: Maintains existing behavior for standard platforms while enabling new functionality for custom backends
- **Numerically verified**: Implementation passes orthogonality and reconstruction tests
- **Configurable**: Can be forced to use composite operations via environment variable `JAX_QR_USE_COMPOSITE=1`

## Implementation Details

The implementation adds composite attributes to QR-related operations by patching the `custom_call` function to insert the appropriate `stablehlo.composite` attribute for QR operations. Key attributes:
- `qr.geqrf` for the QR factorization operation
- `qr.householder_product` for the orthogonal matrix generation

## Testing

The implementation includes comprehensive test coverage:
- Unit tests verifying composite attributes are correctly applied
- Numerical verification ensuring Q is orthogonal and Q*R = original matrix 
- Cross-platform testing with both default settings and forced composite mode

## Performance Impact

No performance impact for standard platforms as the existing implementation path is preserved. Custom backends gain the ability to recognize and implement QR operations explicitly.

## Documentation

Added inline documentation for all components describing:
- How the composite system works
- When composite vs. custom_call paths are taken
- Edge cases and error handling

## Future Work

This implementation could serve as a template for adding composite operations to other linear algebra functions in JAX, providing a consistent pattern for extending JAX's capabilities to custom backends.
