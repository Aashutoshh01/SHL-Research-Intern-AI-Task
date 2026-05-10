"""Recall@10 Evaluation using realistic hiring traces."""
from .client import AsyncEvalClient
from .models import TraceResult
from .metrics import compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_hit_at_k

# Define a robust set of traces covering diverse industries and roles
TRACES = [
    {
        "name": "Java Engineer",
        "messages": [{"role": "user", "content": "I need to assess a senior Java developer. Looking for Spring and Core Java skills."}],
        "expected": ["core java (advanced level) (new)", "spring", "java frameworks"]
    },
    {
        "name": "Financial Analyst",
        "messages": [{"role": "user", "content": "Hiring graduate financial analysts - final-year students, no work experience. We need numerical reasoning and a finance knowledge test."}],
        "expected": ["financial accounting (new)", "basic statistics (new)", "numerical reasoning"]
    },
    {
        "name": "Contact Centre Agent",
        "messages": [{"role": "user", "content": "Screen contact centre agents for English US calls."}],
        "expected": ["svar - spoken english (us) (new)", "contact center call simulation"]
    },
    {
        "name": "Safety Plant Operator",
        "messages": [{"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety and dependability are critical."}],
        "expected": ["dependability and safety instrument (dsi)", "workplace health and safety"]
    },
    {
        "name": "Admin Assistant",
        "messages": [{"role": "user", "content": "Screen admin assistants. Must know Word and Excel."}],
        "expected": ["ms excel (new)", "ms word"]
    },
    {
        "name": "VP of Engineering",
        "messages": [{"role": "user", "content": "Hiring a new VP of Engineering. Need a comprehensive leadership and personality profile."}],
        "expected": ["opq leadership report", "opq universal competency", "occupational personality questionnaire opq32r"]
    },
    {
        "name": "Sales Manager",
        "messages": [{"role": "user", "content": "Looking for assessments for B2B sales managers."}],
        "expected": ["sales", "opq mq sales report"]
    },
    {
        "name": "Data Scientist",
        "messages": [{"role": "user", "content": "Need tests for a Senior Data Scientist involving statistics and python."}],
        "expected": ["data science (new)", "basic statistics (new)", "python"]
    },
    {
        "name": "Multi-Turn Refinement (Sales to Marketing)",
        "messages": [
            {"role": "user", "content": "I am hiring for a sales role."},
            {"role": "assistant", "content": "I recommend the OPQ Sales report and B2B sales skills tests. Can I clarify the seniority?"},
            {"role": "user", "content": "Actually, change the role to Marketing Manager. We need leadership and marketing skills."}
        ],
        "expected": ["marketing (new)", "occupational personality questionnaire opq32r", "OPQ Leadership Report"]
    },
    {
        "name": "Multi-Turn Refinement (Add Personality)",
        "messages": [
            {"role": "user", "content": "Need tests for a software developer."},
            {"role": "assistant", "content": "I recommend Java, Python, and C++ tests."},
            {"role": "user", "content": "Add a personality assessment to check team fit."}
        ],
        "expected": ["occupational personality questionnaire opq32r", "opq team impact"]
    }
]

class RecallEvaluator:
    def __init__(self, client: AsyncEvalClient):
        self.client = client

    async def run_all(self) -> list[TraceResult]:
        results = []
        for trace in TRACES:
            resp, latency = await self.client.chat(trace["messages"])
            
            returned_names = []
            if resp:
                returned_names = [r.get("name", "") for r in resp.get("recommendations", [])]
            
            # Find missing
            returned_lower = [x.lower() for x in returned_names]
            missing = [exp for exp in trace["expected"] if not any(exp.lower() in r or r in exp.lower() for r in returned_lower)]
            
            res = TraceResult(
                trace_name=trace["name"],
                recall_10=compute_recall_at_k(trace["expected"], returned_names, 10),
                recall_5=compute_recall_at_k(trace["expected"], returned_names, 5),
                precision_5=compute_precision_at_k(trace["expected"], returned_names, 5),
                mrr=compute_mrr(trace["expected"], returned_names),
                hit_at_k=compute_hit_at_k(trace["expected"], returned_names, 10),
                expected=trace["expected"],
                returned=returned_names,
                missing=missing,
                latency_ms=latency
            )
            results.append(res)
        return results
