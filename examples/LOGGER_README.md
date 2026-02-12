# Logger Usage Guide

The logging system provides flexible output control for optimization algorithms in the tomocam library.

## Logger Modes

The `Logger` class supports three modes:

### 1. **STDOUT** (Default)
Logs messages to standard output (console).

```cpp
#include "logger.h"

using namespace tomocam;

Logger logger(LogMode::STDOUT);
logger.log("Message to console\n");
```

### 2. **FILE**
Logs messages to a specified file.

```cpp
Logger logger(LogMode::FILE, "optimization.log");
logger.log("Message to file\n");
```

### 3. **SILENT**
Suppresses all log output.

```cpp
Logger logger(LogMode::SILENT);
logger.log("This will not be displayed\n");
```

## Using Logger with NAG Optimization

The `nagopt` function accepts an optional `Logger*` parameter (default is `nullptr`, which uses STDOUT).

### Example 1: Default behavior (STDOUT)

```cpp
#include "optimize.h"

using namespace tomocam::opt;

// Uses default STDOUT logging
auto result = nagopt(grad, loss, x, max_iters, lipschitz, tol, xtol, max_inner_iters);
```

### Example 2: Log to file

```cpp
#include "logger.h"
#include "optimize.h"

using namespace tomocam;
using namespace tomocam::opt;

Logger logger(LogMode::FILE, "optimization_output.log");
auto result = nagopt(grad, loss, x, max_iters, lipschitz, tol, xtol, max_inner_iters, &logger);
```

### Example 3: Silent mode (suppress output)

```cpp
#include "logger.h"
#include "optimize.h"

using namespace tomocam;
using namespace tomocam::opt;

Logger logger(LogMode::SILENT);
auto result = nagopt(grad, loss, x, max_iters, lipschitz, tol, xtol, max_inner_iters, &logger);
```

## Improved Log Messages

The updated logging system provides more informative messages:

### Initialization
```
NAG Optimization started: max_iters=100, lipschitz=1.5, tol=1.00e-06, xtol=1.00e-04
```

### Iteration Progress
```
Iter   0: loss=1.234567e+02, x-error=5.678901e-02, step=6.666667e-02, inner_iters=3
Iter   1: loss=8.765432e+01, x-error=3.456789e-02, step=6.666667e-02, inner_iters=2
...
```

### Convergence
```
Convergence achieved at iter 42: loss 9.876543e-07 < tol 1.00e-06
```
or
```
Convergence achieved at iter 42: x-error 8.765432e-05 < xtol 1.00e-04
```

## Header Files

Include the following headers to use the logging system:

```cpp
#include "logger.h"     // For Logger class
#include "optimize.h"   // For optimization functions
```

## Notes

- The logger automatically flushes to file after each message to ensure data persistence.
- If a file cannot be opened, the logger automatically falls back to STDOUT with a warning.
- Logger instances manage their own file resources and automatically close files on destruction.
- The default behavior (when no logger is passed) remains unchanged for backward compatibility.
