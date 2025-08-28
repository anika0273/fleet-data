# Airflow Setup for Existing Project

This README provides step-by-step instructions and best practices for integrating Apache Airflow into an existing project using Docker Compose. It aims to simplify starting, running, and troubleshooting Airflow alongside your current services and databases.

---

## Prerequisites

- Docker & Docker Compose installed (Docker Compose v2 recommended)
- Existing project with Postgres and other services running under Docker Compose
- Basic familiarity with Airflow concepts and Docker

---

## File Structure Overview

- `docker-compose.yml`: Defines the Airflow services (`airflow-init`, `airflow-webserver`, `airflow-scheduler`) along with your existing project services
- `init-multiple-db.sh`: Script to create the `airflow` database in the Postgres container
- `dags/`: Directory holding Airflow DAG files, mounted into Airflow containers
- `.env`: Optional file holding environment variables including Postgres connection credentials

---

## Quick Start

1. **Add the `init-multiple-db.sh` script**

   This script creates an `airflow` database for Airflow metadata without interfering with your existing database (`fleet_db`).

2. **Update your Docker Compose**

   - Mount `init-multiple-db.sh` into Postgres service under `/docker-entrypoint-initdb.d/`
   - Make sure your Postgres service uses a named volume for data persistence
   - Configure all Airflow services with a common Fernet key and the Airflow DB connection string pointing to the new `airflow` DB
   - Existing project services continue using the original database

3. **Start your stack fresh to initialize databases**

```bash
docker-compose down
docker volume rm [your_postgres_volume_name] # Removes old data volume
docker-compose up -d
```

4. **Initialize Airflow metadata DB and create the admin user**

```bash
docker-compose run --rm airflow-init
```

5. **Start Airflow webserver and scheduler**

```bash
docker-compose up -d airflow-webserver airflow-scheduler
```

6. **Access the Airflow UI**

Open your browser at [http://localhost:8080](http://localhost:8080)  
Login with the admin credentials (admin/admin) set during initialization.

---

## Important Considerations

- **Volume Persistence and Initialization:**  
  Postgres init scripts run _only once_ on a fresh volume. Changing scripts or adding databases requires either removing the volume or creating the database manually.

- **Database Isolation:**  
  Keep Airflow's metadata database (`airflow`) separate from your existing project database to prevent collisions and data loss.

- **Fernet Key Consistency:**  
  Use the _same_ Fernet key (`AIRFLOW__CORE__FERNET_KEY`) across all Airflow containers to avoid encryption/decryption issues.

- **Airflow Version Compatibility:**  
  Match the Airflow version used in Docker images consistently to avoid runtime or compatibility problems.

- **Resource Availability:**  
  Ensure Docker has enough CPU and memory allocated; Airflow services (scheduler, webserver) and your other services might be resource-intensive.

- **Environment Variables:**  
  Share common environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, DB connection strings) through `.env` files or centralized docker-compose config sections.

- **Avoid Using Deprecated Commands:**  
  Use `airflow db migrate` instead of deprecated `airflow db init` for database migrations going forward.

---

## Common Troubleshooting

| Symptom                          | Cause                                      | Solution                                              |
| -------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| Airflow web UI not loading       | Webserver container not running or crashes | Check logs, run `docker-compose up airflow-webserver` |
| Airflow DB connection failure    | `airflow` DB missing or network issues     | Create DB manually or via init script, check network  |
| Database init script not running | Volume already initialized                 | Remove volume or create DB manually                   |
| Encryption/decryption errors     | Fernet keys mismatch                       | Use the exact same Fernet key in all containers       |

---

## Helpful Commands

- List volumes: `docker volume ls`
- Remove volume: `docker volume rm <volume_name>`
- Check Postgres DBs:  
  `docker exec -it fleet-data-postgres psql -U postgres -c "\l"`
- View logs:  
  `docker-compose logs -f airflow-webserver`  
  `docker-compose logs -f airflow-scheduler`
- Run Airflow DB check inside container:  
  `docker exec -it fleet-data-airflow-webserver airflow db check`

---

## References

- [Official Airflow Docker Compose Guide](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- Docker Postgres Init Scripts Behavior
- Airflow CLI Command Reference
