#!/usr/bin/env python
"""
Test career pathway diversity across 5 distinct resume profiles.
Run: python test_diversity.py
"""
import sys
sys.path.insert(0, str(__file__.replace("test_diversity.py", "")))

from api.services.skill_extractor import extract_skills_from_text
from api.services.matcher import load_job_index, embed_query
from api.services.pathways import match_pathways, build_pathway_index

RESUMES = {
    "Data Scientist": """
    Jane Doe
    Senior Data Scientist | 5 years experience
    
    TECHNICAL SKILLS
    Python, R, SQL, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM
    Machine Learning, Deep Learning, NLP, Computer Vision, Time Series Forecasting
    Statistics, A/B Testing, Causal Inference, Experiment Design
    MLflow, Kubeflow, Airflow, Docker, Kubernetes
    AWS SageMaker, GCP Vertex AI, Databricks
    Tableau, Power BI, Matplotlib, Seaborn, Plotly
    Git, Jupyter, VS Code, Linux
    
    EXPERIENCE
    Senior Data Scientist @ TechCorp (2021-Present)
    - Built end-to-end ML pipelines for customer churn prediction (AUC 0.92)
    - Developed NLP models for sentiment analysis on 1M+ reviews
    - Deployed models on Kubernetes with MLflow tracking
    - Led A/B testing framework for recommendation engine
    
    Data Scientist @ StartupXYZ (2019-2021)
    - Built predictive models for fraud detection saving $2M/year
    - Created automated feature engineering pipelines
    - Visualized insights for executive stakeholders
    
    EDUCATION
    M.S. Computer Science, Stanford University
    B.S. Mathematics, UC Berkeley
    """,

    "Frontend Developer": """
    Alex Chen
    Senior Frontend Engineer | 6 years experience
    
    TECHNICAL SKILLS
    TypeScript, JavaScript (ES6+), React, Next.js, Redux, React Query, Zustand
    HTML5, CSS3, Tailwind CSS, Styled Components, Sass, CSS Modules
    Webpack, Vite, Turbopack, ESLint, Prettier, Jest, React Testing Library, Cypress
    REST APIs, GraphQL, Apollo Client, TanStack Query, SWR
    Node.js, Express, NestJS (backend basics)
    Git, GitHub Actions, Vercel, Netlify, Docker
    Figma, Storybook, Chromatic, Accessibility (WCAG)
    
    EXPERIENCE
    Senior Frontend Engineer @ WebScale (2021-Present)
    - Architected micro-frontend architecture for 50+ developer team
    - Built design system used across 15 products
    - Optimized Core Web Vitals: LCP < 1.5s, CLS < 0.1
    - Mentored 5 junior engineers on React patterns
    
    Frontend Developer @ AgencyCo (2018-2021)
    - Delivered 20+ client projects using React/Next.js
    - Implemented pixel-perfect UIs from Figma designs
    - Set up CI/CD pipelines with GitHub Actions
    
    EDUCATION
    B.S. Computer Science, University of Washington
    """,

    "Backend Developer (Java/Spring)": """
    Marcus Johnson
    Senior Backend Engineer | 7 years experience
    
    TECHNICAL SKILLS
    Java 17/21, Spring Boot 3.x, Spring Cloud, Spring Security, Spring Data JPA
    Kotlin, Groovy, Maven, Gradle
    PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
    Kafka, RabbitMQ, gRPC, Protocol Buffers
    Docker, Kubernetes, Helm, ArgoCD
    AWS (EC2, RDS, S3, Lambda, ECS), Terraform
    JUnit, Mockito, Testcontainers, ArchUnit
    Git, GitLab CI/CD, SonarQube, Prometheus, Grafana
    Microservices, Domain-Driven Design, Event Sourcing, CQRS
    
    EXPERIENCE
    Senior Backend Engineer @ FinTech Inc (2020-Present)
    - Designed high-throughput payment processing system (10K TPS)
    - Migrated monolith to 12 microservices on Kubernetes
    - Implemented event-driven architecture with Kafka
    - Reduced p99 latency from 500ms to 80ms
    
    Backend Developer @ EnterpriseSoft (2017-2020)
    - Built REST APIs for CRM platform serving 500K users
    - Optimized SQL queries, reduced DB load by 60%
    - Introduced contract testing with Pact
    
    EDUCATION
    M.S. Software Engineering, Carnegie Mellon
    B.S. Computer Engineering, Georgia Tech
    """,

    "DevOps / Cloud Engineer": """
    Sarah Williams
    Senior DevOps Engineer | 6 years experience
    
    TECHNICAL SKILLS
    AWS (EC2, VPC, EKS, ECS, RDS, S3, CloudFront, IAM, Lambda, CloudFormation)
    Terraform, Terragrunt, Pulumi, CrossPlane
    Kubernetes, Helm, Kustomize, Operators, Istio, Linkerd
    Docker, Podman, Buildah, Kaniko
    CI/CD: GitHub Actions, GitLab CI, Jenkins, ArgoCD, Flux, Spinnaker
    Linux, Bash, Python, Go (for operators/tools)
    Prometheus, Grafana, Loki, Tempo, Alertmanager, Thanos
    ELK Stack, Datadog, New Relic
    HashiCorp Vault, Consul, Boundary
    GitOps, Infrastructure as Code, Policy as Code (OPA)
    
    EXPERIENCE
    Senior DevOps Engineer @ CloudNative Co (2021-Present)
    - Built self-service developer platform on EKS for 200 engineers
    - Implemented GitOps with ArgoCD across 50+ clusters
    - Designed multi-region DR strategy (RPO < 1min, RTO < 15min)
    - Reduced cloud spend 35% via rightsizing and spot instances
    
    DevOps Engineer @ ScaleUp (2018-2021)
    - Migrated 100+ services from VMs to Kubernetes
    - Built CI/CD pipelines deploying 50x/day
    - Implemented comprehensive observability stack
    
    CERTIFICATIONS
    AWS Solutions Architect Professional, CKA, CKAD, Terraform Associate
    
    EDUCATION
    B.S. Information Systems, NYU
    """,

    "Data Engineer": """
    David Kim
    Senior Data Engineer | 5 years experience
    
    TECHNICAL SKILLS
    Python, SQL, Scala, PySpark, Apache Spark, Delta Lake
    Airflow, Dagster, Prefect, dbt, SQLMesh
    Snowflake, BigQuery, Redshift, Databricks, ClickHouse
    Kafka, Kafka Connect, Debezium, Flink, RisingWave
    Docker, Kubernetes, Terraform, AWS/GCP/Azure
    Git, GitHub Actions, GitLab CI, Great Expectations, Soda
    Data Modeling: Kimball, Data Vault, One Big Table
    Parquet, Avro, ORC, Iceberg, Hudi
    
    EXPERIENCE
    Senior Data Engineer @ DataDriven Inc (2021-Present)
    - Built petabyte-scale data lake on S3 + Iceberg + Trino
    - Designed 200+ dbt models with automated testing
    - Implemented CDC pipelines with Debezium + Kafka
    - Reduced data freshness from 24h to 15min
    
    Data Engineer @ AnalyticsCo (2019-2021)
    - Migrated on-prem Hadoop to cloud (Snowflake + dbt)
    - Built real-time streaming pipeline for fraud alerts
    - Established data contracts and schema registry
    
    EDUCATION
    M.S. Data Science, Columbia University
    B.S. Computer Science, University of Michigan
    """,
}


def main():
    print("Loading job index...")
    index = load_job_index()
    pathway_index = build_pathway_index(index)
    print(f"Index loaded: {len(index.ids)} jobs, {len(pathway_index.skill_sets)} pathways\n")

    for name, text in RESUMES.items():
        print(f"{'='*60}")
        print(f"RESUME: {name}")
        print(f"{'='*60}")

        skills = extract_skills_from_text(text)
        print(f"Extracted skills ({len(skills)}): {', '.join(skills[:15])}{'...' if len(skills) > 15 else ''}")

        query_vec = embed_query(skills, "", resume_context=text[:300])

        ranked = match_pathways(
            query_vec,
            top_k=len(pathway_index.skill_sets),  # all pathways
            index=index,
            pathway_index=pathway_index,
            skills=skills,
            exclude_fallback=True,
        )

        # Apply same filtering as views.py
        filtered = [
            pw for pw in ranked
            if pw.score >= 0.30 and pw.coverage >= 0.15
        ]
        if len(filtered) < 3:
            visible = ranked[:3]
        else:
            visible = filtered[:8]
        additional = [pw for pw in ranked if pw not in visible]

        print(f"\nVisible pathways ({len(visible)}):")
        for i, pw in enumerate(visible, 1):
            matched = pw.matched_skills
            print(f"  {i}. {pw.name:<22} | score={pw.score:.3f} | coverage={pw.coverage:.2f} | matched={len(matched)} skills")
            if matched:
                print(f"      -> {', '.join(matched[:8])}{'...' if len(matched) > 8 else ''}")

        print(f"\nAdditional (Show more) ({len(additional)}):")
        for pw in additional:
            print(f"  - {pw.name:<22} | score={pw.score:.3f} | coverage={pw.coverage:.2f}")

        print()


if __name__ == "__main__":
    main()