/* Example usage of the Logger class with NAG optimization
 * 
 * This example demonstrates three logging modes:
 * 1. STDOUT - logs to console (default)
 * 2. FILE - logs to a file
 * 3. SILENT - suppresses all output
 */

#include "logger.h"
#include "optimize.h"
#include "array.h"
#include <iostream>

using namespace tomocam;
using namespace tomocam::opt;

int main() {
    // Example 1: Default logging to STDOUT
    std::cout << "=== Example 1: Logging to STDOUT ===\n";
    Logger stdout_logger(LogMode::STDOUT);
    stdout_logger.log("This message goes to console\n");
    
    // Example 2: Logging to a file
    std::cout << "\n=== Example 2: Logging to FILE ===\n";
    Logger file_logger(LogMode::FILE, "optimization.log");
    file_logger.log("This message goes to optimization.log\n");
    std::cout << "Check optimization.log for output\n";
    
    // Example 3: Silent mode (no output)
    std::cout << "\n=== Example 3: SILENT mode ===\n";
    Logger silent_logger(LogMode::SILENT);
    silent_logger.log("This message is suppressed\n");
    std::cout << "No log message should appear above\n";
    
    std::cout << "\n=== Usage with nagopt ===\n";
    std::cout << "To use with nagopt function:\n";
    std::cout << "  // Stdout (default)\n";
    std::cout << "  auto result = nagopt(grad, loss, x, max_iters, lip, tol, xtol, inner_iters);\n";
    std::cout << "  \n";
    std::cout << "  // Custom logger\n";
    std::cout << "  Logger logger(LogMode::FILE, \"output.log\");\n";
    std::cout << "  auto result = nagopt(grad, loss, x, max_iters, lip, tol, xtol, inner_iters, &logger);\n";
    std::cout << "  \n";
    std::cout << "  // Silent mode\n";
    std::cout << "  Logger logger(LogMode::SILENT);\n";
    std::cout << "  auto result = nagopt(grad, loss, x, max_iters, lip, tol, xtol, inner_iters, &logger);\n";
    
    return 0;
}
