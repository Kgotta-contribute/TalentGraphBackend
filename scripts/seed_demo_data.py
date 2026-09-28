import asyncio
import uuid
from datetime import datetime
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.mandate import Mandate
from app.models.candidate import Candidate
from app.models.evaluation import CandidateEvaluation

# Mandate A: Nexus Intelligence Labs
MANDATE_A_ID = uuid.UUID("a0000000-0000-0000-0000-000000000001")
# Mandate B: Aether Cloud Networks
MANDATE_B_ID = uuid.UUID("b0000000-0000-0000-0000-000000000002")

async def seed():
    async with AsyncSessionLocal() as session:
        # Check if already seeded
        m_check = await session.execute(select(Mandate).where(Mandate.id == MANDATE_A_ID))
        if m_check.scalar_one_or_none():
            print("Demo mandates already exist. Refreshing evaluations...")
        else:
            # Mandate A
            mandate_a = Mandate(
                id=MANDATE_A_ID,
                recruiter_id="dev-recruiter",
                title="Senior Full-Stack & AI Systems Engineer",
                company="Nexus Intelligence Labs",
                raw_jd="""We are seeking a Senior Full-Stack & AI Systems Engineer to architect and build our next-generation multi-agent LLM platform.

Key Responsibilities:
- Architect and maintain asynchronous Python/FastAPI microservices.
- Build multi-agent LLM & RAG retrieval systems with LangGraph/LangChain.
- Develop responsive React & TypeScript interfaces.
- Optimize vector database indexes and semantic search retrieval (FAISS, pgvector).
- Deploy containerized services to cloud infrastructure with CI/CD.

Mandatory Technical Skills:
Python, FastAPI, LangGraph, LangChain, React, TypeScript, Vector Search, FAISS, Docker, PostgreSQL.

Preferred Qualifications:
AWS, Kubernetes, Redis, Celery, Sentence-Transformers, ChromaDB, CI/CD, Tailwind CSS.

Experience Target:
4+ Years (Senior). Bachelor's or Master's in Computer Science or related quantitative field.""",
                status="active",
                job_requirements={
                    "role": "Senior Full-Stack & AI Systems Engineer",
                    "experience_target_years": 4,
                    "education_criteria": "Bachelor's or Master's in Computer Science or related quantitative field",
                    "mandatory_skills": [
                        "Python", "FastAPI", "LangGraph", "LangChain", "React",
                        "TypeScript", "Vector Search", "FAISS", "Docker", "PostgreSQL"
                    ],
                    "preferred_skills": [
                        "AWS", "Kubernetes", "Redis", "Celery",
                        "Sentence-Transformers", "ChromaDB", "CI/CD", "Tailwind CSS"
                    ],
                    "soft_skills": [
                        "Problem Solving", "System Architecture",
                        "Cross-functional Collaboration", "Technical Communication"
                    ],
                    "responsibilities": [
                        "Architect and maintain asynchronous Python/FastAPI microservices",
                        "Build multi-agent LLM & RAG retrieval systems with LangGraph/LangChain",
                        "Develop responsive React & TypeScript interfaces",
                        "Optimize vector database indexes and semantic search retrieval",
                        "Deploy containerized services to cloud infrastructure with CI/CD"
                    ],
                    "domain_tags": ["AI/ML", "Full-Stack", "Distributed Systems", "Cloud", "Vector Search"]
                }
            )
            session.add(mandate_a)

            # Mandate B
            mandate_b = Mandate(
                id=MANDATE_B_ID,
                recruiter_id="dev-recruiter",
                title="Senior Cloud Platform & DevOps Engineer",
                company="Aether Cloud Networks",
                raw_jd="""Aether Cloud Networks is hiring a Senior Cloud Platform & DevOps Engineer to lead infrastructure automation, Kubernetes orchestration, and multi-cloud resilience.

Key Requirements:
- Deep expertise in Kubernetes cluster architecture, Helm charts, and service meshes.
- Advanced Infrastructure as Code using Terraform and cloud automation.
- Production Linux systems engineering and kernel-level performance tuning.
- Multi-cloud execution across AWS and GCP with zero-downtime CI/CD.

Mandatory Skills:
Kubernetes, Terraform, Linux, Helm, Docker, CI/CD, Python, AWS.

Preferred Skills:
ArgoCD, Prometheus, Grafana, Golang, GCP, PostgreSQL.""",
                status="active",
                job_requirements={
                    "role": "Senior Cloud Platform & DevOps Engineer",
                    "experience_target_years": 5,
                    "education_criteria": "B.S. in Computer Science, Software Engineering, or equivalent",
                    "mandatory_skills": [
                        "Kubernetes", "Terraform", "Linux", "Helm",
                        "Docker", "CI/CD", "Python", "AWS"
                    ],
                    "preferred_skills": [
                        "ArgoCD", "Prometheus", "Grafana", "Golang", "GCP", "PostgreSQL"
                    ],
                    "soft_skills": [
                        "Site Reliability Engineering", "Incident Management",
                        "Infrastructure Architecture", "Team Mentorship"
                    ],
                    "responsibilities": [
                        "Architect and maintain production EKS/GKE Kubernetes clusters",
                        "Automate multi-region infrastructure provisioning using Terraform",
                        "Implement GitOps pipelines with ArgoCD and automated canary releases",
                        "Monitor real-time system metrics with Prometheus and Grafana dashboards"
                    ],
                    "domain_tags": ["Cloud", "DevOps", "Kubernetes", "Infrastructure as Code", "SRE"]
                }
            )
            session.add(mandate_b)
            await session.commit()

        # Seed candidates
        candidates_data = [
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000001"),
                "full_name": "Sarah Chen",
                "email": "sarah.chen@techmail.io",
                "current_title": "Senior AI Full-Stack Engineer",
                "location": "San Francisco, CA",
                "years_experience": "5.5",
                "github_url": "https://github.com/sarahchen-ai",
                "skills": ["Python", "FastAPI", "LangGraph", "LangChain", "React", "TypeScript", "FAISS", "Docker", "PostgreSQL", "AWS", "Kubernetes", "Redis", "Vector Search"],
                "summary": "Senior Full-Stack & AI Engineer with 5.5 years of experience building scalable distributed microservices, multi-agent LLM systems, and high-performance React frontends.",
                "experience": [
                    {"company": "Vanguard Cognitive Inc.", "title": "Senior AI Full-Stack Engineer", "start": "2022", "end": "Present", "description": "Architected asynchronous Python/FastAPI microservices and multi-agent LangGraph orchestrations handling 200k daily requests. Integrated FAISS & pgvector indexes with 99.4% recall.", "tech_tags": ["Python", "FastAPI", "LangGraph", "React", "TypeScript", "FAISS", "PostgreSQL"]},
                    {"company": "CloudPulse Systems", "title": "Full Stack Software Engineer", "start": "2019", "end": "2022", "description": "Developed React/TypeScript dashboards and containerized backend services deployed via Docker and Kubernetes.", "tech_tags": ["React", "TypeScript", "Docker", "Kubernetes", "AWS"]}
                ],
                "projects": [
                    {"name": "Enterprise Agentic RAG Platform", "description": "High-throughput multi-agent retrieval system using LangGraph, FAISS vector index, and FastAPI backend.", "tech_tags": ["FastAPI", "LangGraph", "FAISS", "React", "TypeScript", "Docker"]},
                    {"name": "Real-Time Telemetry Microservices", "description": "Distributed event streaming microservices with Redis caching and PostgreSQL persistence.", "tech_tags": ["Python", "Redis", "PostgreSQL", "Docker", "AWS"]}
                ],
                "education": [{"degree": "B.S. in Computer Science", "institution": "University of California, Berkeley", "graduation_year": 2019}],
                "certifications": ["AWS Certified Solutions Architect – Associate", "DeepLearning.AI LangChain Specialist"],
                "score": 94.6,
                "tier": "Strongly Recommended",
                "tech_pct": 100.0,
                "score_tech": 100.0, "score_exp": 96.0, "score_jd": 94.0, "score_proj": 92.0, "score_edu": 88.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000002"),
                "full_name": "Lucas Silva",
                "email": "lucas.silva@raglabs.dev",
                "current_title": "Applied AI & RAG Engineer",
                "location": "Seattle, WA",
                "years_experience": "4.5",
                "github_url": "https://github.com/lucassilva-rag",
                "skills": ["Python", "FastAPI", "LangGraph", "LangChain", "Vector Search", "PostgreSQL", "Docker", "TypeScript", "React", "FAISS"],
                "summary": "Specialized in agentic pipelines, RAG semantic search, and asynchronous API architectures.",
                "experience": [
                    {"company": "Aether AI Labs", "title": "Applied AI Engineer", "start": "2021", "end": "Present", "description": "Engineered autonomous LLM agents with LangGraph and vector search.", "tech_tags": ["Python", "LangGraph", "FastAPI", "pgvector"]}
                ],
                "projects": [{"name": "VectorSearch Engine", "description": "Sub-millisecond FAISS vector search engine.", "tech_tags": ["FAISS", "Python", "Docker"]}],
                "education": [{"degree": "B.S. in Software Engineering", "institution": "University of Washington", "graduation_year": 2020}],
                "certifications": ["TensorFlow Developer Certificate"],
                "score": 92.7,
                "tier": "Strongly Recommended",
                "tech_pct": 100.0,
                "score_tech": 98.0, "score_exp": 92.0, "score_jd": 93.0, "score_proj": 88.0, "score_edu": 85.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000003"),
                "full_name": "Priya Patel",
                "email": "priya.patel@cloudengine.io",
                "current_title": "Senior MLOps & Cloud Engineer",
                "location": "Austin, TX",
                "years_experience": "6.0",
                "github_url": "https://github.com/priyapatel-cloud",
                "skills": ["Kubernetes", "Docker", "Python", "FastAPI", "AWS", "PostgreSQL", "CI/CD", "Terraform"],
                "summary": "Cloud platform engineer with 6 years leading Kubernetes orchestration and automated deployments.",
                "experience": [
                    {"company": "DataScale Tech", "title": "Senior MLOps Engineer", "start": "2020", "end": "Present", "description": "Built automated container delivery pipelines using Kubernetes, Docker, and AWS.", "tech_tags": ["Kubernetes", "Docker", "AWS", "Python"]}
                ],
                "projects": [{"name": "K8s Model Deployment Mesh", "description": "Automated deployment controller for machine learning services.", "tech_tags": ["Kubernetes", "Python", "Docker"]}],
                "education": [{"degree": "M.S. in Computer Engineering", "institution": "UT Austin", "graduation_year": 2018}],
                "certifications": ["CKA: Certified Kubernetes Administrator"],
                "score": 84.5,
                "tier": "Recommended",
                "tech_pct": 80.0,
                "score_tech": 85.0, "score_exp": 92.0, "score_jd": 82.0, "score_proj": 80.0, "score_edu": 90.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000004"),
                "full_name": "Marcus Vance",
                "email": "marcus.vance@archsystems.net",
                "current_title": "Principal Backend Architect",
                "location": "Chicago, IL",
                "years_experience": "9.0",
                "github_url": "https://github.com/marcusvance",
                "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Redis", "Distributed Systems"],
                "summary": "Deep backend architect with 9 years designing high-throughput database systems and event-driven microservices.",
                "experience": [
                    {"company": "FinTech Core", "title": "Principal Architect", "start": "2018", "end": "Present", "description": "Led backend services processing 1B+ monthly transactions in Python & PostgreSQL.", "tech_tags": ["Python", "PostgreSQL", "AWS"]}
                ],
                "projects": [{"name": "Distributed Event Engine", "description": "High concurrency event router with sub-second latency.", "tech_tags": ["Python", "Redis", "Docker"]}],
                "education": [{"degree": "B.S. in Computer Science", "institution": "University of Illinois", "graduation_year": 2015}],
                "certifications": ["AWS Solutions Architect Professional"],
                "score": 81.1,
                "tier": "Recommended",
                "tech_pct": 70.0,
                "score_tech": 80.0, "score_exp": 100.0, "score_jd": 78.0, "score_proj": 75.0, "score_edu": 85.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000005"),
                "full_name": "Elena Rostova",
                "email": "elena.rostova@mlresearch.org",
                "current_title": "Machine Learning Engineer",
                "location": "Boston, MA",
                "years_experience": "4.0",
                "github_url": "https://github.com/erostova",
                "skills": ["Python", "LangChain", "Vector Search", "Sentence-Transformers", "FastAPI", "Docker"],
                "summary": "ML engineer focused on NLP, embedding models, and transformer architectures.",
                "experience": [
                    {"company": "Cognitive Scale", "title": "ML Engineer", "start": "2021", "end": "Present", "description": "Fine-tuned embedding models and integrated RAG pipelines.", "tech_tags": ["Python", "Sentence-Transformers", "Docker"]}
                ],
                "projects": [{"name": "EmbeddingBenchmark", "description": "Comparative evaluation of open-source embedding models.", "tech_tags": ["Python", "PyTorch"]}],
                "education": [{"degree": "M.S. in Data Science", "institution": "Northeastern University", "graduation_year": 2021}],
                "certifications": ["Deep Learning Specialization"],
                "score": 75.4,
                "tier": "Recommended",
                "tech_pct": 70.0,
                "score_tech": 75.0, "score_exp": 80.0, "score_jd": 76.0, "score_proj": 74.0, "score_edu": 82.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000006"),
                "full_name": "David Kim",
                "email": "david.kim@codebase.io",
                "current_title": "Junior Python Developer",
                "location": "San Jose, CA",
                "years_experience": "2.5",
                "github_url": "https://github.com/davidkim-dev",
                "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
                "summary": "Early-career developer with foundational experience in Python backend APIs and Docker containers.",
                "experience": [
                    {"company": "StartupLab", "title": "Junior Developer", "start": "2023", "end": "Present", "description": "Built REST endpoints and maintained PostgreSQL schemas.", "tech_tags": ["Python", "FastAPI", "PostgreSQL"]}
                ],
                "projects": [{"name": "Task Tracker API", "description": "CRUD API with JWT authentication.", "tech_tags": ["Python", "FastAPI"]}],
                "education": [{"degree": "B.S. in Information Systems", "institution": "San Jose State University", "graduation_year": 2023}],
                "certifications": [],
                "score": 49.6,
                "tier": "Weak Match",
                "tech_pct": 31.0,
                "score_tech": 45.0, "score_exp": 52.0, "score_jd": 50.0, "score_proj": 48.0, "score_edu": 65.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000007"),
                "full_name": "Jonathan Taylor",
                "email": "jonathan.taylor@enterprise.corp",
                "current_title": "Lead Enterprise Systems Developer",
                "location": "Dallas, TX",
                "years_experience": "12.0",
                "github_url": "https://github.com/jtaylor-systems",
                "skills": ["Java", "Spring Boot", "Oracle SQL", "Linux", "Docker", "RESTful APIs"],
                "summary": "Extensive enterprise developer transitioning to modern Python and cloud services.",
                "experience": [
                    {"company": "Legacy Corp", "title": "Lead Systems Developer", "start": "2013", "end": "Present", "description": "Maintained monolithic enterprise systems in Java and Oracle SQL.", "tech_tags": ["Java", "Oracle", "Linux"]}
                ],
                "projects": [{"name": "Legacy Migration Gateway", "description": "REST gateway bridging legacy Java monolith to microservices.", "tech_tags": ["Java", "Docker"]}],
                "education": [{"degree": "B.S. in Computer Information Systems", "institution": "Texas A&M University", "graduation_year": 2012}],
                "certifications": ["Oracle Certified Professional"],
                "score": 43.2,
                "tier": "Weak Match",
                "tech_pct": 25.0,
                "score_tech": 38.0, "score_exp": 70.0, "score_jd": 42.0, "score_proj": 40.0, "score_edu": 75.0
            },
            {
                "id": uuid.UUID("c0000000-0000-0000-0000-000000000008"),
                "full_name": "Amina Al-Mansoor",
                "email": "amina.mansoor@frontendcraft.design",
                "current_title": "Senior Frontend Developer",
                "location": "Palo Alto, CA",
                "years_experience": "5.0",
                "github_url": "https://github.com/amina-ui",
                "skills": ["React", "TypeScript", "Next.js", "Tailwind CSS", "HTML5/CSS3", "JavaScript"],
                "summary": "Specialist in high-fidelity React component design, UI performance optimization, and accessible design systems.",
                "experience": [
                    {"company": "DesignFirst Interactive", "title": "Senior Frontend Engineer", "start": "2020", "end": "Present", "description": "Built component design systems and complex data dashboards in React and TypeScript.", "tech_tags": ["React", "TypeScript", "Tailwind CSS"]}
                ],
                "projects": [{"name": "Accessible UI Component Kit", "description": "Open-source WCAG-compliant design system.", "tech_tags": ["React", "TypeScript"]}],
                "education": [{"degree": "B.A. in Interactive Media & CS", "institution": "Stanford University", "graduation_year": 2020}],
                "certifications": [],
                "score": 40.2,
                "tier": "Weak Match",
                "tech_pct": 13.0,
                "score_tech": 35.0, "score_exp": 75.0, "score_jd": 38.0, "score_proj": 40.0, "score_edu": 70.0
            }
        ]

        for cand in candidates_data:
            # Check candidate
            c_res = await session.execute(select(Candidate).where(Candidate.id == cand["id"]))
            c_existing = c_res.scalar_one_or_none()
            if not c_existing:
                c_obj = Candidate(
                    id=cand["id"],
                    full_name=cand["full_name"],
                    email=cand["email"],
                    current_title=cand["current_title"],
                    years_experience=cand["years_experience"],
                    github_url=cand["github_url"],
                    profile={
                        "full_name": cand["full_name"],
                        "email": cand["email"],
                        "location": cand["location"],
                        "current_title": cand["current_title"],
                        "years_experience": float(cand["years_experience"]),
                        "skills": cand["skills"],
                        "summary": cand["summary"],
                        "experience_history": cand["experience"],
                        "projects": cand["projects"],
                        "education": cand["education"],
                        "certifications": cand["certifications"],
                        "github_url": cand["github_url"]
                    }
                )
                session.add(c_obj)

            # Check evaluation for Mandate A
            e_res = await session.execute(
                select(CandidateEvaluation).where(
                    CandidateEvaluation.candidate_id == cand["id"],
                    CandidateEvaluation.mandate_id == MANDATE_A_ID
                )
            )
            e_existing = e_res.scalar_one_or_none()
            if not e_existing:
                matched_req = [s for s in cand["skills"] if s in ["Python", "FastAPI", "LangGraph", "LangChain", "React", "TypeScript", "Vector Search", "FAISS", "Docker", "PostgreSQL"]]
                gaps = [s for s in ["Python", "FastAPI", "LangGraph", "LangChain", "React", "TypeScript", "Vector Search", "FAISS", "Docker", "PostgreSQL"] if s not in cand["skills"]]

                eval_obj = CandidateEvaluation(
                    mandate_id=MANDATE_A_ID,
                    candidate_id=cand["id"],
                    tech_coverage_pct=cand["tech_pct"],
                    score_technical=cand["score_tech"],
                    score_experience=cand["score_exp"],
                    score_jd_similarity=cand["score_jd"],
                    score_projects=cand["score_proj"],
                    score_education=cand["score_edu"],
                    final_score=cand["score"],
                    tier=cand["tier"],
                    github_verified=True,
                    status="ranked",
                    verification_result={
                        "candidate_id": str(cand["id"]),
                        "matched_required": matched_req,
                        "matched_preferred": [s for s in cand["skills"] if s in ["AWS", "Kubernetes", "Redis", "Celery", "Sentence-Transformers", "ChromaDB", "CI/CD", "Tailwind CSS"]],
                        "required_gaps": gaps,
                        "preferred_gaps": [s for s in ["AWS", "Kubernetes", "Redis", "Celery", "Sentence-Transformers", "ChromaDB", "CI/CD", "Tailwind CSS"] if s not in cand["skills"]],
                        "tech_coverage_pct": cand["tech_pct"],
                        "evidence": [
                            {"requirement": s, "status": "MATCHED", "source": "RESUME", "evidence": f"Demonstrated in professional work and projects as {cand['current_title']}", "confidence": 0.9}
                            for s in matched_req
                        ]
                    },
                    report={
                        "candidate_id": str(cand["id"]),
                        "executive_summary": f"{cand['full_name']} is a {cand['current_title']} with {cand['years_experience']} years of experience. Demonstrated strong technical proficiency across core requirements with {cand['tech_pct']}% tech coverage.",
                        "key_strengths": [
                            f"Direct technical mastery in {', '.join(cand['skills'][:4])}",
                            f"{cand['years_experience']} years hands-on production engineering",
                            "Demonstrated track record of delivering end-to-end architectures",
                            "Verified credentials and open-source portfolio"
                        ],
                        "identified_skill_gaps": gaps,
                        "risk_factors": [f"Requires ramp-up on {', '.join(gaps[:3])}"] if gaps else ["Low risk profile; strong alignment with mandatory tech stack."],
                        "ramp_up_considerations": ["Review team GitOps and continuous delivery standards", "Familiarization with proprietary internal schemas"],
                        "final_verdict": f"{cand['full_name']} is {cand['tier'].lower()} for this role based on multi-factor evaluation.",
                        "hiring_confidence": round(cand["score"] / 100.0 * 0.92, 2),
                        "relevant_experience": cand["experience"],
                        "relevant_projects": cand["projects"],
                        "github_summary": f"Active GitHub contributor at {cand['github_url']} with verified repositories.",
                        "interview_questions": [
                            {
                                "focus_area": "ARCHITECTURE & PRODUCTION CONCURRENCY",
                                "question": "Can you walk us through the architecture of a high-concurrency production service you built? What were the primary bottlenecks and how did you resolve them?",
                                "rationale": "Validates architectural depth and production-scale engineering judgment.",
                                "keywords": ["throughput", "latency", "caching", "concurrency", "indexes", "trade-offs"]
                            },
                            {
                                "focus_area": "SKILL TRANSITION & LEARNING VELOCITY",
                                "question": f"How would you approach ramping up quickly on technologies like {gaps[0] if gaps else 'next-gen agentic frameworks'} in a fast-paced team?",
                                "rationale": "Tests adaptability and speed of technical onboarding.",
                                "keywords": ["fundamentals", "transferable concepts", "hands-on labs", "rapid prototyping"]
                            },
                            {
                                "focus_area": "PROJECT RETROSPECTIVE & ENGINEERING MATURITY",
                                "question": "Looking back at your most significant project, what technical trade-offs did you make and what would you design differently today?",
                                "rationale": "Assesses engineering maturity, self-reflection, and long-term architectural foresight.",
                                "keywords": ["trade-offs", "maintainability", "scalability", "tech debt", "monitoring"]
                            }
                        ]
                    }
                )
                session.add(eval_obj)

        # Mandate B Candidate Evaluations (Exact values from User Screenshots 2 & 3)
        mandate_b_scores = {
            "Sarah Chen": {
                "score": 75.6, "tier": "Recommended", "tech_pct": 63.0,
                "matched_req": ["Kubernetes", "AWS", "Docker", "CI/CD", "Python"],
                "gaps_req": ["Terraform", "Linux", "Helm"],
                "matched_pref": ["PostgreSQL"],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP"],
                "score_tech": 76.0, "score_exp": 78.0, "score_jd": 74.0, "score_proj": 72.0, "score_edu": 80.0
            },
            "Marcus Vance": {
                "score": 75.4, "tier": "Recommended", "tech_pct": 63.0,
                "matched_req": ["Python", "Docker", "AWS", "Linux", "CI/CD"],
                "gaps_req": ["Kubernetes", "Terraform", "Helm"],
                "matched_pref": ["PostgreSQL"],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP"],
                "score_tech": 75.0, "score_exp": 90.0, "score_jd": 72.0, "score_proj": 70.0, "score_edu": 78.0
            },
            "Elena Rostova": {
                "score": 62.9, "tier": "Consider for Interview", "tech_pct": 44.0,
                "matched_req": ["Python", "Docker"],
                "gaps_req": ["Kubernetes", "Terraform", "Linux", "Helm", "CI/CD", "AWS"],
                "matched_pref": [],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP", "PostgreSQL"],
                "score_tech": 58.0, "score_exp": 72.0, "score_jd": 64.0, "score_proj": 65.0, "score_edu": 75.0
            },
            "Lucas Silva": {
                "score": 60.1, "tier": "Consider for Interview", "tech_pct": 31.0,
                "matched_req": ["Python", "Docker"],
                "gaps_req": ["Kubernetes", "Terraform", "Linux", "Helm", "CI/CD", "AWS"],
                "matched_pref": ["PostgreSQL"],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP"],
                "score_tech": 55.0, "score_exp": 70.0, "score_jd": 62.0, "score_proj": 64.0, "score_edu": 72.0
            },
            "David Kim": {
                "score": 49.6, "tier": "Weak Match", "tech_pct": 31.0,
                "matched_req": ["Python", "Docker"],
                "gaps_req": ["Kubernetes", "Terraform", "Linux", "Helm", "CI/CD", "AWS"],
                "matched_pref": ["PostgreSQL"],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP"],
                "score_tech": 45.0, "score_exp": 52.0, "score_jd": 50.0, "score_proj": 48.0, "score_edu": 65.0
            },
            "Jonathan Taylor": {
                "score": 43.2, "tier": "Weak Match", "tech_pct": 25.0,
                "matched_req": ["Linux", "Docker"],
                "gaps_req": ["Kubernetes", "Terraform", "Helm", "CI/CD", "Python", "AWS"],
                "matched_pref": [],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP", "PostgreSQL"],
                "score_tech": 38.0, "score_exp": 70.0, "score_jd": 42.0, "score_proj": 40.0, "score_edu": 75.0
            },
            "Amina Al-Mansoor": {
                "score": 40.2, "tier": "Weak Match", "tech_pct": 13.0,
                "matched_req": ["CI/CD"],
                "gaps_req": ["Kubernetes", "Terraform", "Linux", "Helm", "Docker", "Python", "AWS"],
                "matched_pref": [],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP", "PostgreSQL"],
                "score_tech": 35.0, "score_exp": 75.0, "score_jd": 38.0, "score_proj": 40.0, "score_edu": 70.0
            },
            "Priya Patel": {
                "score": 88.5, "tier": "Strongly Recommended", "tech_pct": 88.0,
                "matched_req": ["Kubernetes", "Terraform", "Docker", "CI/CD", "Python", "AWS"],
                "gaps_req": ["Helm", "Linux"],
                "matched_pref": ["PostgreSQL"],
                "gaps_pref": ["ArgoCD", "Prometheus", "Grafana", "Golang", "GCP"],
                "score_tech": 90.0, "score_exp": 92.0, "score_jd": 86.0, "score_proj": 84.0, "score_edu": 88.0
            }
        }

        for cand in candidates_data:
            b_eval = await session.execute(
                select(CandidateEvaluation).where(
                    CandidateEvaluation.candidate_id == cand["id"],
                    CandidateEvaluation.mandate_id == MANDATE_B_ID
                )
            )
            b_existing = b_eval.scalar_one_or_none()
            if not b_existing:
                score_data = mandate_b_scores.get(cand["full_name"], {
                    "score": 50.0, "tier": "Weak Match", "tech_pct": 40.0,
                    "matched_req": ["Python", "Docker"],
                    "gaps_req": ["Kubernetes", "Terraform"],
                    "matched_pref": [], "gaps_pref": ["ArgoCD", "Prometheus"],
                    "score_tech": 50.0, "score_exp": 50.0, "score_jd": 50.0, "score_proj": 50.0, "score_edu": 50.0
                })
                b_eval_obj = CandidateEvaluation(
                    mandate_id=MANDATE_B_ID,
                    candidate_id=cand["id"],
                    tech_coverage_pct=score_data["tech_pct"],
                    score_technical=score_data["score_tech"],
                    score_experience=score_data["score_exp"],
                    score_jd_similarity=score_data["score_jd"],
                    score_projects=score_data["score_proj"],
                    score_education=score_data["score_edu"],
                    final_score=score_data["score"],
                    tier=score_data["tier"],
                    github_verified=True,
                    status="ranked",
                    verification_result={
                        "candidate_id": str(cand["id"]),
                        "matched_required": score_data["matched_req"],
                        "matched_preferred": score_data["matched_pref"],
                        "required_gaps": score_data["gaps_req"],
                        "preferred_gaps": score_data["gaps_pref"],
                        "tech_coverage_pct": score_data["tech_pct"],
                        "evidence": [
                            {"requirement": s, "status": "MATCHED", "source": "RESUME", "evidence": f"Demonstrated in infrastructure and platform engineering work", "confidence": 0.9}
                            for s in score_data["matched_req"]
                        ]
                    },
                    report={
                        "candidate_id": str(cand["id"]),
                        "executive_summary": f"{cand['full_name']} evaluated for Senior Cloud Platform & DevOps Engineer. Final score {score_data['score']}% with {score_data['tech_pct']}% technical coverage on Kubernetes and cloud automation.",
                        "key_strengths": [
                            f"Demonstrated competence in {', '.join(score_data['matched_req'][:3])}",
                            f"{cand['years_experience']} years production software experience",
                            "Containerization and microservices architecture background"
                        ],
                        "identified_skill_gaps": score_data["gaps_req"],
                        "risk_factors": [f"Gap in {', '.join(score_data['gaps_req'][:2])} - requires onboarding ramp-up"],
                        "ramp_up_considerations": ["Review infrastructure as code conventions", "Assess cluster security guardrails"],
                        "final_verdict": f"{cand['full_name']} is categorized as {score_data['tier']} for this DevOps mandate.",
                        "hiring_confidence": round(score_data["score"] / 100.0 * 0.92, 2),
                        "relevant_experience": cand["experience"],
                        "relevant_projects": cand["projects"],
                        "github_summary": f"Verified GitHub activity at {cand['github_url']}",
                        "interview_questions": [
                            {
                                "focus_area": "KUBERNETES & CLOUD RESILIENCE",
                                "question": "How do you approach multi-region failover and stateful workloads in Kubernetes?",
                                "rationale": "Directly probes core infrastructure and high availability experience.",
                                "keywords": ["Kubernetes", "failover", "multi-region", "SRE", "RTO", "RPO"]
                            }
                        ]
                    }
                )
                session.add(b_eval_obj)

        await session.commit()
        print("Demo data seeded successfully with 2 mandates and 8 candidates!")

if __name__ == "__main__":
    asyncio.run(seed())
