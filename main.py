import loading_workflow as workflow
LLM_ENDPOINT = "http://127.0.0.1:8000/generate"

# ML-backend(ов) с Whisper для транскрипции.
# Список: можно указать один или несколько URL.
# Каждый — это отдельный экземпляр MLBackend/main.py на своём порту.
# Задачи распределяются по ним round-robin, семафор на каждый.
ML_BACKEND_URLS = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
]

TARGET_TOPIC = "Python программирование"
BATCH_SIZE = 10
DOWNLOAD_THRESHOLD = 6
MANIFEST_PATH = "download_manifest.json"
TRANSCRIBE_PER_BACKEND_CONCURRENT = 2
TRANSCRIBE_TIMEOUT = 300

def main():
    loader, raw_courses = workflow.fetch_stepik_courses(
        topic=TARGET_TOPIC, 
        limit=100
    )
    
    if not raw_courses:
        return


    analyzed_results = workflow.analyze_courses_relevance(
        raw_courses=raw_courses,
        topic=TARGET_TOPIC,
        llm_endpoint=LLM_ENDPOINT,
        batch_size=BATCH_SIZE
    )

    workflow.print_top_results(analyzed_results, top_n=20)


    manifest_path = workflow.plan_download_manifest(
        loader=loader,
        analyzed_courses=analyzed_results,
        raw_courses=raw_courses,
        min_score=DOWNLOAD_THRESHOLD,
        topic=TARGET_TOPIC,
        llm_endpoint=LLM_ENDPOINT,
        manifest_path=MANIFEST_PATH
    )


    workflow.execute_downloads(
        loader=loader,
        raw_courses=raw_courses,
        manifest_path=manifest_path,
        ml_backend_urls=ML_BACKEND_URLS,
        transcribe_per_backend_concurrent=TRANSCRIBE_PER_BACKEND_CONCURRENT,
        transcribe_timeout=TRANSCRIBE_TIMEOUT,
    )

if __name__ == "__main__":
    main()