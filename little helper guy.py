import os
import asyncio
import inspect
import re
import json
from google import genai
from google.generativeai import GenerativeModel, configure

class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, callable] = {}

    def register(self, name: str, fn: callable):
        self.tools[name] = fn

    async def execute(self, name: str, inp):
        fn = self.tools.get(name)
        if fn is None:
            raise ValueError(f"No tool registered under '{name}'")
        # If it's async, await it; otherwise offload to threadpool
        if inspect.iscoroutinefunction(fn):
            return await fn(inp)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, inp)

# --- Tools ---
async def list_files(path: str) -> list[str]:
    """Return a list of filenames in path."""
    return os.listdir(path)

async def read_files(path: str) -> dict[str, str]:
    """Read all files under path; return {filename: content}."""
    result = {}
    for fname in os.listdir(path):
        file_path = os.path.join(path, fname)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                result[fname] = f.read()
    return result

async def write_files(data: dict[str, str]) -> str:
    """Write multiple files; data is {filename: content}."""
    for fname, content in data.items():
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(content)
    return "OK"

class AgenticAI:
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model_name = "gemini-2.5-flash-preview-04-17"
        self.tool_registry = ToolRegistry()
        # register our tools
        self.tool_registry.register("list_files", list_files)
        self.tool_registry.register("read_files", read_files)
        self.tool_registry.register("write_files", write_files)

    def _clean_json(self, text: str) -> str:
        # strip markdown fences, etc.
        return re.sub(r"```(?:json)?|```", "", text).strip()

    def generate_plan(self, request: str) -> list[dict]:
        prompt = f"You are an autonomous agent. Plan steps (as JSON) to satisfy: {request}"
        response = GenerativeModel(self.model_name).generate_content([prompt])
        raw = self._clean_json(response.text)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(f"Could not parse plan JSON: {raw!r}")

    async def _execute_step(self, step: dict) -> dict:
        tool_name = step["tool"]
        inp = step.get("input")
        result = await self.tool_registry.execute(tool_name, inp)
        return {"step": step, "result": result}

    async def execute_plan(self, plan: list[dict]) -> list[dict]:
        # dispatch all steps concurrently
        tasks = [self._execute_step(step) for step in plan]
        return await asyncio.gather(*tasks)

    def _synthesize_followup(self, history: list[dict]) -> str:
        # generate the next request based on history
        summary = json.dumps(history, indent=2)
        prompt = f"History:\n{summary}\n\nWhat is the next action to fully satisfy the original request?"
        response = GenerativeModel(self.model_name).generate_content([prompt])
        return response.text.strip()

    async def _run(self, request: str, max_iterations: int = 10) -> list[dict]:
        history = []
        current = request
        for _ in range(max_iterations):
            plan = self.generate_plan(current)
            results = await self.execute_plan(plan)
            history.extend(results)

            if any(r["step"]["tool"] == "finish" for r in results):
                break

            current = self._synthesize_followup(history)

        return history

    def run(self, request: str, max_iterations: int = 10) -> list[dict]:
        # entry point: run the async loop
        return asyncio.run(self._run(request, max_iterations))

def main():
    agent = AgenticAI(api_key="AIzaSyCUZaott1f-ES2PZES7CZeh1pxYSbN74wg")
    result = agent.run("Fetch all .txt files in ./docs and summarize them")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
