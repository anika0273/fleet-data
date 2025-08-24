# Java Setup and Running The Spark ETL Pipeline on macOS (Detailed Guide)

## Why Java 17?

Apache Spark (versions 3.5+ and many others) requires Java 17 or later to run properly due to JVM compatibility.
Java 16 or below causes errors such as:

- java.lang.UnsupportedClassVersionError
- Spark failing to launch or connect
- Network bind errors on macOS

## How I Upgraded Java and Configured the Environment

- ### Step 1: Install Java 17 via Homebrew

```bash
brew install openjdk@17
```

This installs the latest OpenJDK 17 runtime.

- ### Step 2: Make Java 17 Discoverable by macOS

```bash
sudo ln -sfn /usr/local/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
```

Symbolic link allows macOS and Java tools to find the new JDK.

- ### Step 3: Set Java Environment Variables

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v17)
export PATH=$JAVA_HOME/bin:$PATH
```

This makes the shell use Java 17 executables in the current session.

You can add these lines to ~/.zshrc or ~/.bash_profile for persistence.

- ### Step 4: Verify Your Java Installation

```bash
java -version
```

Output should confirm Java 17, like:

```text
openjdk version "17.0.x" ...
```

- ### Step 5: Create and Activate Python Virtual Environment

```bash
python3.11 -m venv fleet_venv
source fleet_venv/bin/activate
```

Isolates your Python dependencies per project to avoid conflicts.

- ### Step 6: Install PySpark

```bash
pip install pyspark
```

Provides Spark API for Python.

- ### Step 7: Add Java and Python to Firewall Exceptions (macOS specific)

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home/bin/java
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/fleet_venv/bin/python
```

Prevents macOS firewall from blocking Spark’s local network communication, which is essential for distributed processes on your machine.

- ### Step 8: Fix Spark’s Local IP Binding Issue on macOS

```bash
export SPARK_LOCAL_IP=$(ipconfig getifaddr en0)
```

Without this, Spark sometimes binds only to localhost, causing hostname resolution issues.

- ### Step 9: Run Your PySpark ETL Script
  Inside your Python virtual environment, execute:

```bash
python batch_etl/batch_etl.py
```

This runs your batch ETL pipeline: reads from PostgreSQL, cleans data, aggregates metrics, writes cleaned data back to PostgreSQL and local Parquet.

## How The Utilities (utils.py) Work

- get_spark_session()
  Creates a SparkSession object configured with the PostgreSQL JDBC driver specified by the JAR in your lib folder.
  This session is used across the ETL for distributed Spark DataFrame operations, enabling reading/writing to PostgreSQL.

- get_jdbc_properties()
  Returns the connection URL along with username, password, and driver class information —
  centralizing database credentials so they aren’t hardcoded multiple times.

These abstractions simplify your ETL (batch_etl.py) by separating environment setup from business logic.

# Summary of Commands

```bash

brew install openjdk@17
sudo ln -sfn /usr/local/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
export JAVA_HOME=$(/usr/libexec/java_home -v17)
export PATH=$JAVA_HOME/bin:$PATH
java -version

python3.11 -m venv fleet_venv
source fleet_venv/bin/activate
pip install pyspark

sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home/bin/java
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/fleet_venv/bin/python

export SPARK_LOCAL_IP=$(ipconfig getifaddr en0)
python batch_etl/batch_etl.py
```
