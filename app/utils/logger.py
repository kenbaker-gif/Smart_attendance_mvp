import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys # Added for system-level printing if needed

# --- 1. DEFINE PATHS AND CREATE DIRECTORY ---

# Use the Current Working Directory (CWD) to define the logs path.
# This assumes you run your application (e.g., 'uvicorn app.main:app') from the project root.
LOG_DIR = Path(os.getcwd()) / "logs"

# Print the directory being used for verification to the standard error stream (console)
# This print statement helps confirm the path is correct before logging initializes.
print(f"--- LOG DIRECTORY BEING USED: {LOG_DIR.resolve()} ---", file=sys.stderr) 

# Create logs directory if missing
try:
    # Use parents=True to create intermediate directories if necessary
    LOG_DIR.mkdir(exist_ok=True, parents=True) 
except Exception as e:
    # Print a raw error if directory creation fails, as logging isn't fully set up yet.
    print(f"ERROR: Could not create log directory at {LOG_DIR}: {e}", file=sys.stderr)

LOG_FILE = LOG_DIR / "attendance.log"


# --- 2. CONFIGURE LOGGER AND LEVEL ---

# Configure main logger
logger = logging.getLogger("attendance_system")

# TEMPORARY FIX: Set to DEBUG to capture ALL messages for troubleshooting.
# Change back to logging.INFO once file logging is confirmed working.
logger.setLevel(logging.DEBUG) 


# --- 3. DEFINE HANDLERS ---

# File Handler (rotates when 5MB)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8"
)
# TEMPORARY FIX: Match file handler level to logger's debug level
file_handler.setLevel(logging.DEBUG) 

# Console Handler
console_handler = logging.StreamHandler()
# TEMPORARY FIX: Match console handler level to logger's debug level
console_handler.setLevel(logging.DEBUG) 


# --- 4. DEFINE FORMATTER AND ATTACH ---

# Format: Includes %(name)s for clarity
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initial log to confirm successful setup (should appear in console and file now)
logger.info("Logging configuration loaded successfully.")

# Note: Any code calling logger.debug() will now be written to the file as well.