import asyncio
import uuid
from datetime import datetime
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.mandate import Mandate
from app.models.candidate import Candidate
from app.models.evaluation import CandidateEvaluation

TEMPLATES_DATA = [
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000001"),
        "title": "Senior Full-Stack & AI Systems Engineer",
        "company": "Nexus Intelligence Labs",
        "raw_jd": """We are seeking a Senior Full-Stack & AI Systems Engineer to architect and build our next-generation multi-agent LLM platform.

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
        "job_requirements": {
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
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000002"),
        "title": "Senior Cloud Platform & DevOps Engineer",
        "company": "Aether Cloud Networks",
        "raw_jd": """Aether Cloud Networks is hiring a Senior Cloud Platform & DevOps Engineer to lead infrastructure automation, Kubernetes orchestration, and multi-cloud resilience.

Key Requirements:
- Deep expertise in Kubernetes cluster architecture, Helm charts, and service meshes.
- Advanced Infrastructure as Code using Terraform and cloud automation.
- Production Linux systems engineering and kernel-level performance tuning.
- Multi-cloud execution across AWS and GCP with zero-downtime CI/CD.

Mandatory Skills:
Kubernetes, Terraform, Linux, Helm, Docker, CI/CD, Python, AWS.

Preferred Skills:
ArgoCD, Prometheus, Grafana, Golang, GCP, PostgreSQL.""",
        "job_requirements": {
            "role": "Senior Cloud Platform & DevOps Engineer",
            "experience_target_years": 5,
            "education_criteria": "B.S. in Computer Science, Software Engineering, or equivalent operational experience",
            "mandatory_skills": [
                "Kubernetes", "Terraform", "Linux", "Helm",
                "Docker", "CI/CD", "Python", "AWS"
            ],
            "preferred_skills": [
                "ArgoCD", "Prometheus", "Grafana", "Golang", "GCP", "PostgreSQL"
            ],
            "soft_skills": [
                "Incident Management", "Reliability Engineering",
                "DevOps Culture", "Infrastructure Architecture"
            ],
            "responsibilities": [
                "Architect and maintain production EKS/GKE Kubernetes clusters",
                "Automate multi-region infrastructure provisioning using Terraform",
                "Implement GitOps pipelines with ArgoCD and automated canary releases",
                "Monitor real-time system metrics with Prometheus and Grafana dashboards"
            ],
            "domain_tags": ["Cloud", "DevOps", "Kubernetes", "Infrastructure as Code", "SRE"]
        }
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000003"),
        "title": "Staff Machine Learning Engineer",
        "company": "Cognitive Core AI",
        "raw_jd": """Cognitive Core AI is seeking a Staff Machine Learning Engineer to lead foundation model training, multi-agent LLM systems, and distributed GPU inference.

Key Requirements:
- 7+ years building and deploying large-scale Machine Learning systems in production.
- Deep expertise in PyTorch, Transformer architectures, and LLM fine-tuning techniques (LoRA, QLoRA).
- Hands-on experience with multi-agent orchestration frameworks (LangGraph, AutoGen) and RAG pipelines.
- Distributed training across GPU clusters (DeepSpeed, Megatron-LM).

Preferred Qualifications:
- Published papers at top-tier ML conferences (NeurIPS, ICML, ICLR).
- Experience with high-performance C++ extensions and TensorRT-LLM inference optimization.
- Ph.D. or M.S. in Computer Science, Artificial Intelligence, or Electrical Engineering.""",
        "job_requirements": {
            "role": "Staff Machine Learning Engineer",
            "experience_target_years": 7,
            "education_criteria": "Ph.D. or M.S. in Computer Science, Artificial Intelligence, or Electrical Engineering",
            "mandatory_skills": [
                "Python", "PyTorch", "Transformers", "LLMs",
                "LangGraph", "DeepSpeed", "Distributed Training", "CUDA"
            ],
            "preferred_skills": [
                "TensorRT-LLM", "C++", "vLLM", "Triton", "Ray", "vLLM", "Vector Search"
            ],
            "soft_skills": [
                "Research Leadership", "AI Ethics & Alignment", "Technical Mentorship"
            ],
            "responsibilities": [
                "Lead research and production deployment of LLM fine-tuning and agentic systems",
                "Optimize distributed multi-node GPU clusters for high-throughput inference",
                "Architect state-of-the-art multi-agent RAG pipelines"
            ],
            "domain_tags": ["Machine Learning", "LLMs", "PyTorch", "Distributed Training", "AI Research"]
        }
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000004"),
        "title": "Software Development Engineer II",
        "company": "Amazon",
        "raw_jd": """AMAZON Software Dev Engineer II Paragon 

About the job:
Would you like to work on one of the world's largest transactional distributed systems? SPS is looking for software engineers who thrive on complex problems and solve for operating complex and mission critical systems under high loads.

Key job responsibilities:
- Take ownership over the software design, documentation, development, and support of systems built natively in AWS.
- Collaborate with leaders, work backwards from customers, identify problems, propose innovative solutions.

Basic Qualifications:
- 1+ years of non-internship professional software development experience
- Experience programming with at least one software programming language (Java, C++, Python)
- AWS Native Systems & Cloud Architecture
- High-load Transactional Distributed Systems

Preferred Qualifications:
- Bachelor's degree in computer science or equivalent
- Experience with distributed multi-tier systems and RESTful web services""",
        "job_requirements": {
            "role": "Software Development Engineer II",
            "experience_target_years": 1,
            "education_criteria": "Bachelor's degree in Computer Science or equivalent",
            "mandatory_skills": [
                "Java", "Python", "AWS Native Systems", "Cloud Architecture",
                "Distributed Systems", "RESTful APIs", "Software Development Lifecycle"
            ],
            "preferred_skills": [
                "DynamoDB", "SQS", "SNS", "Lambda", "Microservices", "Docker"
            ],
            "soft_skills": [
                "Customer Obsession", "Ownership", "Bias for Action", "Deliver Results"
            ],
            "responsibilities": [
                "Design and operate high-scale transactional distributed systems",
                "Develop cloud-native services natively in AWS",
                "Write clean, robust, and maintainable unit and integration tests"
            ],
            "domain_tags": ["Distributed Systems", "Cloud", "AWS", "Backend", "E-Commerce"]
        }
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000005"),
        "title": "Software Engineer II (AI Governance)",
        "company": "Rippling",
        "raw_jd": """Rippling
Software Engineer II (AI Governance)

About the job:
The AI Governance team sits at the intersection of identity, access control, model routing, MCP security, data protection, spend management, and auditability. The goal is to give companies one governed path for AI usage across employees, agents, models, and business tools.

What You Will Do:
- Build high-quality, reliable AI governance products with meticulous attention to detail.
- Contribute to the design and implementation of MCP access controls, model gateway policies, runtime authorization, and audit pipelines.
- Use AI-native development practices to increase engineering velocity.

What You Will Need:
- 3+ years of software engineering experience building and operating production systems.
- Strong software engineering fundamentals (Python, Go, Django, React, MongoDB, AWS).
- Sound security and systems judgment around identity, authorization, or policy enforcement.
- Familiarity with AI infrastructure, model gateways, Model Context Protocol (MCP), and data protection.""",
        "job_requirements": {
            "role": "Software Engineer II (AI Governance)",
            "experience_target_years": 3,
            "education_criteria": "Bachelor's or Master's degree in Computer Science or equivalent",
            "mandatory_skills": [
                "Python", "Go", "Django", "React", "AWS",
                "AI-native Engineering", "Security & Systems Judgment", "API Development"
            ],
            "preferred_skills": [
                "Model Context Protocol (MCP)", "Model Gateways", "AI Infrastructure",
                "Identity & Access Management", "Data Protection", "MongoDB"
            ],
            "soft_skills": [
                "High Agency", "Product Ownership", "Bias Toward Shipping", "Collaboration"
            ],
            "responsibilities": [
                "Design runtime authorization and audit pipelines for LLM agents",
                "Implement MCP access control filters and model gateway proxies",
                "Build policy-aware budget controls for enterprise AI consumption"
            ],
            "domain_tags": ["AI Governance", "Security", "MCP", "Full-Stack", "Identity"]
        }
    }
]

async def seed_templates():
    async with AsyncSessionLocal() as session:
        # Fetch candidates
        c_res = await session.execute(select(Candidate))
        candidates = c_res.scalars().all()
        print(f"Found {len(candidates)} existing candidates in database.")

        for tpl in TEMPLATES_DATA:
            m_res = await session.execute(select(Mandate).where(Mandate.id == tpl["id"]))
            existing_m = m_res.scalar_one_or_none()
            if not existing_m:
                m = Mandate(
                    id=tpl["id"],
                    recruiter_id="dev-recruiter",
                    title=tpl["title"],
                    company=tpl["company"],
                    raw_jd=tpl["raw_jd"],
                    status="active",
                    job_requirements=tpl["job_requirements"]
                )
                session.add(m)
                print(f"Created template mandate: {tpl['title']} ({tpl['company']})")
            else:
                existing_m.title = tpl["title"]
                existing_m.company = tpl["company"]
                existing_m.raw_jd = tpl["raw_jd"]
                existing_m.job_requirements = tpl["job_requirements"]
                print(f"Updated template mandate: {tpl['title']}")

            await session.commit()

            # Seed evaluations for candidates if missing
            mandate_id = tpl["id"]
            for cand in candidates:
                e_res = await session.execute(
                    select(CandidateEvaluation).where(
                        CandidateEvaluation.mandate_id == mandate_id,
                        CandidateEvaluation.candidate_id == cand.id
                    )
                )
                if not e_res.scalar_one_or_none():
                    cand_skills = cand.profile.get("skills", []) if cand.profile else []
                    mand_skills = tpl["job_requirements"]["mandatory_skills"]
                    pref_skills = tpl["job_requirements"]["preferred_skills"]

                    matched_req = [s for s in cand_skills if any(m.lower() in s.lower() or s.lower() in m.lower() for m in mand_skills)]
                    req_gaps = [s for s in mand_skills if not any(c.lower() in s.lower() or s.lower() in c.lower() for c in cand_skills)]
                    matched_pref = [s for s in cand_skills if any(p.lower() in s.lower() or s.lower() in p.lower() for p in pref_skills)]

                    tech_pct = min(100.0, round((len(matched_req) / max(1, len(mand_skills))) * 100, 1))
                    score_tech = min(100.0, round(tech_pct * 0.95 + 5.0, 1))
                    try:
                        cand_yrs = float(cand.years_experience) if cand.years_experience and cand.years_experience != 'None' else 4.0
                    except (ValueError, TypeError):
                        cand_yrs = 4.0
                    score_exp = min(100.0, round(cand_yrs / max(1, tpl["job_requirements"]["experience_target_years"]) * 85.0, 1))
                    score_jd = round(tech_pct * 0.9, 1)
                    score_proj = 80.0
                    score_edu = 85.0

                    final_score = round(score_tech * 0.40 + score_exp * 0.25 + score_jd * 0.20 + score_proj * 0.10 + score_edu * 0.05, 1)
                    if final_score >= 90:
                        tier = "Strongly Recommended"
                    elif final_score >= 75:
                        tier = "Recommended"
                    elif final_score >= 60:
                        tier = "Consider for Interview"
                    elif final_score >= 40:
                        tier = "Weak Match"
                    else:
                        tier = "Not Recommended"

                    evaluation = CandidateEvaluation(
                        mandate_id=mandate_id,
                        candidate_id=cand.id,
                        tech_coverage_pct=tech_pct,
                        score_technical=score_tech,
                        score_experience=score_exp,
                        score_jd_similarity=score_jd,
                        score_projects=score_proj,
                        score_education=score_edu,
                        final_score=final_score,
                        tier=tier,
                        github_verified=bool(cand.github_url),
                        status="ranked",
                        verification_result={
                            "candidate_id": str(cand.id),
                            "matched_required": matched_req,
                            "matched_preferred": matched_pref,
                            "required_gaps": req_gaps,
                            "preferred_gaps": [s for s in pref_skills if s not in matched_pref],
                            "tech_coverage_pct": tech_pct,
                            "evidence": [
                                {"requirement": s, "status": "MATCHED", "source": "RESUME", "evidence": f"Demonstrated in background as {cand.current_title}", "confidence": 0.9}
                                for s in matched_req
                            ]
                        },
                        report={
                            "candidate_id": str(cand.id),
                            "executive_summary": f"{cand.full_name} is a {cand.current_title} evaluated for {tpl['title']} at {tpl['company']}. Final score {final_score}% with {tech_pct}% tech coverage.",
                            "key_strengths": [
                                f"Direct match in {', '.join(matched_req[:3])}" if matched_req else "Strong foundational software engineering skills",
                                f"{cand.years_experience} years of hands-on production experience",
                                "Demonstrated engineering ownership and clean delivery track record"
                            ],
                            "identified_skill_gaps": req_gaps[:4],
                            "risk_factors": [f"Gap in {', '.join(req_gaps[:2])} - requires onboarding ramp"] if req_gaps else ["No critical skill gaps identified."],
                            "ramp_up_considerations": ["Review domain-specific team architecture and standards"],
                            "final_verdict": f"{cand.full_name} is categorized as {tier} for this role.",
                            "hiring_confidence": round(final_score / 100.0 * 0.92, 2),
                            "relevant_experience": cand.profile.get("experience_history", []) if cand.profile else [],
                            "relevant_projects": cand.profile.get("projects", []) if cand.profile else [],
                            "github_summary": f"Verified GitHub activity at {cand.github_url}" if cand.github_url else "No GitHub portfolio provided.",
                            "interview_questions": [
                                {
                                    "focus_area": "SYSTEM ARCHITECTURE & PRODUCTION SCALE",
                                    "question": f"Can you describe your experience designing scalable systems aligned with {tpl['title']} requirements?",
                                    "rationale": "Directly probes core competencies and technical depth.",
                                    "keywords": ["scalability", "architecture", "trade-offs", "performance"]
                                },
                                {
                                    "focus_area": "TECHNICAL GAPS & RAMP-UP",
                                    "question": f"How would you approach mastering {req_gaps[0] if req_gaps else 'new frameworks'} quickly in a fast-moving sprint?",
                                    "rationale": "Validates self-learning agility and onboarding velocity.",
                                    "keywords": ["learning agility", "documentation", "prototyping"]
                                }
                            ]
                        }
                    )
                    session.add(evaluation)

        await session.commit()
        print("All 5 templates successfully seeded with candidate evaluations!")

if __name__ == "__main__":
    asyncio.run(seed_templates())
