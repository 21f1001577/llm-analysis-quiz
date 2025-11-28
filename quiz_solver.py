# quiz_solver.py
import logging
import time
import requests
import json
import traceback
from browser_handler import BrowserHandler
from data_processor import DataProcessor
from llm_helper import LLMHelper
from urllib.parse import urljoin
import os
from bs4 import BeautifulSoup
import re
import pandas as pd
from typing import Optional, Any

logger = logging.getLogger(__name__)


class QuizSolver:
    """Main class for solving quiz tasks"""

    def __init__(self, config):
        self.config = config
        self.browser = BrowserHandler(config)
        self.data_processor = DataProcessor(config)
        self.llm = LLMHelper(config)
        self.start_time = None
        self.chain_logger = None  # set per chain

    # ===========================================================
    # Logging helpers
    # ===========================================================
    def log_step(self, msg: str):
        logger.info(msg)
        if self.chain_logger:
            self.chain_logger.info(msg)

    def log_json(self, title: str, data: Any):
        try:
            text = f"{title}:\n{json.dumps(data, indent=2)}"
        except Exception:
            text = f"{title}: (unserializable data)"
        logger.info(text)
        if self.chain_logger:
            self.chain_logger.info(text)

    def log_html(self, title: str, html: str):
        snippet = (str(html)[:500]).replace("\n", " ")
        msg = f"{title} (first 500 chars): {snippet} ..."
        logger.info(msg)
        if self.chain_logger:
            self.chain_logger.info(msg)

    def log_prompt(self, title: str, prompt: str, response: Optional[str] = None):
        logger.info(f"=== {title} PROMPT ===\n{prompt}\n=== END PROMPT ===")
        if self.chain_logger:
            self.chain_logger.info(f"=== {title} PROMPT ===\n{prompt}\n=== END PROMPT ===")
        if response is not None:
            logger.info(f"=== {title} RESPONSE ===\n{response}\n=== END RESPONSE ===")
            if self.chain_logger:
                self.chain_logger.info(f"=== {title} RESPONSE ===\n{response}\n=== END RESPONSE ===")

    # ===========================================================
    # Utility helpers
    # ===========================================================
    def _strip_code_fences(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = re.sub(r"(^\s*```(?:json)?\s*)", "", text, flags=re.I)
        text = re.sub(r"(\s*```$)", "", text, flags=re.I)
        return text.strip()
    
    # add near other helpers in quiz_solver.py

    def _serialize_answer(self, answer: Any):
        """
        Ensure answer is an allowed primitive for the remote endpoint.
        - If numeric/bool -> return as-is
        - If str -> return as-is
        - If dict/list -> try common keys ('result','answer','value') to extract primitive
        otherwise return json.dumps(answer) so the endpoint receives a string, not an object.
        """
        # None -> empty string (or could be None depending on server)
        if answer is None:
            return ""

        # Primitive types accepted by server
        if isinstance(answer, (int, float, bool)):
            return answer
        if isinstance(answer, str):
            # sometimes answer may be code-fenced JSON as a string => keep as-is
            return answer

        # For dict-like answers try to extract common primitive fields
        if isinstance(answer, dict):
            for key in ("result", "answer", "value", "result_value", "sum"):
                if key in answer:
                    v = answer[key]
                    # If nested primitive, return it
                    if isinstance(v, (int, float, bool, str)):
                        return v
                    # if nested dict with same keys, fallback to stringifying
            # no common key found or non-primitive value -> stringify
            try:
                return json.dumps(answer)
            except Exception:
                return str(answer)

        # For lists, try to reduce to a comma-joined string (or JSON-stringify)
        if isinstance(answer, (list, tuple)):
            try:
                # if list has a single primitive, return that
                if len(answer) == 1 and isinstance(answer[0], (int, float, bool, str)):
                    return answer[0]
            except Exception:
                pass
            try:
                return json.dumps(answer)
            except Exception:
                return str(answer)

        # Fallback: stringify anything else
        try:
            return json.dumps(answer)
        except Exception:
            return str(answer)


    def _normalize_url(self, base: str, url: Any) -> Optional[str]:
        if not url:
            return None
        try:
            url = str(url).strip()
        except Exception:
            return None
        if not url:
            return None
        return urljoin(base, url)

    def _extract_first_json(self, text: Optional[str]):
        if not text:
            return None
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", str(text), re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
        return None

    # ===========================================================
    # HTML scraping helpers
    # ===========================================================
    def _extract_secret_from_rendered_html(self, html: Optional[str]):
        if not html:
            return None
        try:
            soup = BeautifulSoup(str(html), "html.parser")
            q = soup.select_one("#question")
            text = q.get_text() if q else soup.get_text()
            m = re.search(r"(\d{3,})", text)
            return m.group(1) if m else None
        except Exception:
            return None

    def _find_csv_link_in_rendered_html(self, html: Optional[str], base_url: str):
        if not html:
            return None
        soup = BeautifulSoup(str(html), "html.parser")

        def href_is_csv(h):
            try:
                return bool(h and isinstance(h, str) and h.lower().endswith(".csv"))
            except Exception:
                return False

        a = soup.find("a", href=href_is_csv)
        if a and a.get("href"):
            return self._normalize_url(base_url, a["href"])
        return None

    # ===========================================================
    # CSV processing
    # ===========================================================
    def _compute_sum_from_csv(self, csv_path_or_url: str, cutoff=None, comparator_col=None, value_col=None):
        df = pd.read_csv(csv_path_or_url)
        cols = list(df.columns)
        val_candidates = [c for c in cols if c.lower() == "value"]

        if val_candidates:
            val_col = val_candidates[0]
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric/value column found in CSV")
            val_col = numeric_cols[-1]

        if cutoff is not None:
            comp_col = comparator_col
            if comp_col is None:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                comp_col = next((c for c in numeric_cols if c != val_col), None)

            if comp_col:
                filtered = df[df[comp_col].astype(float) <= float(cutoff)]
            else:
                filtered = df
            s = filtered[val_col].astype(float).sum()
        else:
            s = df[val_col].astype(float).sum()

        if abs(s - int(s)) < 1e-9:
            return int(round(s))
        return float(s)

    # ===========================================================
    # Chain solving
    # ===========================================================
    def solve_quiz_chain(self, url: str, email: str, secret: str):
        os.makedirs("logs", exist_ok=True)
        chain_id = int(time.time())
        chain_log_path = f"logs/chain_{chain_id}.log"

        chain_logger = logging.getLogger(f"CHAIN_{chain_id}")
        chain_logger.setLevel(logging.INFO)
        chain_logger.propagate = False

        fh = logging.FileHandler(chain_log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        chain_logger.addHandler(fh)

        self.chain_logger = chain_logger

        chain_logger.info("=== NEW QUIZ CHAIN STARTED ===")
        chain_logger.info(f"Student Email: {email}")
        chain_logger.info(f"Initial Quiz URL: {url}")
        chain_logger.info(f"Chain Log File: {chain_log_path}")

        self.start_time = time.time()
        current_url = url
        attempt_count = 0
        max_attempts = 10

        while current_url and attempt_count < max_attempts:
            attempt_count += 1
            chain_logger.info(f"Attempt {attempt_count}: Processing {current_url}")

            try:
                elapsed = time.time() - self.start_time
                if elapsed > self.config.MAX_QUIZ_TIME:
                    chain_logger.warning(f"Time limit exceeded: {elapsed:.2f}s")
                    break

                result = self.solve_single_quiz(current_url, email, secret)
                chain_logger.info(f"Result: {result}")

                if result and result.get("url"):
                    current_url = result["url"]
                    chain_logger.info(f"Moving to next quiz: {current_url}")
                else:
                    chain_logger.info("Quiz chain completed")
                    break

            except Exception as e:
                chain_logger.error(f"Error in quiz chain: {e}")
                chain_logger.error(traceback.format_exc())
                break

        try:
            self.browser.close()
        except Exception:
            pass

        for h in list(chain_logger.handlers):
            chain_logger.removeHandler(h)
            h.close()

        self.chain_logger = None

        return {"completed": True, "attempts": attempt_count}

    # ===========================================================
    # Single quiz solving
    # ===========================================================
    def solve_single_quiz(self, quiz_url: Any, email: str, secret: str):
        try:
            quiz_data = self.browser.get_page_content(quiz_url)
            if not quiz_data:
                logger.error("Failed to fetch quiz content")
                return None

            # Normalize dict → use HTML
            quiz_text = ""
            if isinstance(quiz_data, dict):
                quiz_text = quiz_data.get("html") or quiz_data.get("text") or str(quiz_data)
            else:
                quiz_text = str(quiz_data)

            logger.info(f"Quiz content retrieved: {quiz_text[:200]}...")

            # Parse with Mode A (light)
            task_info = self.parse_quiz_content(quiz_text)
            if not task_info:
                logger.error("Failed Mode A parse (no JSON)")
                return None

            task_info["submit_url"] = self._normalize_url(str(quiz_url), task_info.get("submit_url"))
            file_urls = task_info.get("file_urls") or []

            # SAFER URL DISCOVERY
            # Only extract real <a href="..."> links
            soup = BeautifulSoup(quiz_text, "html.parser")
            href_links = [a.get("href") for a in soup.find_all("a") if a.get("href")]

            for u in href_links:
                nu = self._normalize_url(str(quiz_url), u)
                if nu and nu not in file_urls:
                    file_urls.append(nu)

            # Also include URLs inside visible text if they look valid
            visible = soup.get_text(" ")
            matches = re.findall(r'https?://[^\s"<>\']+', visible)
            for u in matches:
                if u not in file_urls:
                    file_urls.append(u)


            # Resolve resource files
            resolved_files = []
            for fu in file_urls:
                full = self._normalize_url(str(quiz_url), fu)
                if not full:
                    continue
                try:
                    head = requests.head(full, allow_redirects=True, timeout=10)
                    ctype = head.headers.get("Content-Type", "").lower()
                except Exception:
                    ctype = ""

                # For HTML → extract secret or CSV
                if "text/html" in ctype or os.path.splitext(full)[1] == "":
                    rendered = self.browser.get_page_content(full)
                    html = rendered.get("html") if isinstance(rendered, dict) else rendered

                    quiz_secret = self._extract_secret_from_rendered_html(html)
                    if quiz_secret:
                        resolved_files.append({"type": "secret", "value": quiz_secret, "source": full})
                        continue

                    csv_link = self._find_csv_link_in_rendered_html(html, full)
                    if csv_link:
                        resolved_files.append({"type": "csv", "url": csv_link})
                        continue

                    resolved_files.append({"type": "html", "html": str(html), "source": full})

                else:
                    ext = os.path.splitext(full)[1].lower()
                    if ext == ".csv" or "text/csv" in ctype:
                        resolved_files.append({"type": "csv", "url": full})
                    elif "pdf" in ctype or ext == ".pdf":
                        resolved_files.append({"type": "pdf", "url": full})
                    elif "audio" in ctype or ext in [".mp3", ".wav", ".opus"]:
                        resolved_files.append({"type": "audio", "url": full})
                    else:
                        resolved_files.append({"type": "download", "url": full})

            logger.info(f"Task: {task_info.get('task')}")
            logger.info(f"Submit URL: {task_info.get('submit_url')}")

            # Solve task
            answer = self.solve_task(task_info, resolved_files)
            if answer is None:
                return None
            if isinstance(answer, str):
                answer = self._strip_code_fences(answer)

            logger.info(f"Generated answer: {answer}")

            # Submit once (Mode A answer)
            result = self.submit_answer(
                task_info.get("submit_url"),
                email,
                secret,
                str(quiz_url),
                answer
            )

            # ===========================================================
            # Hybrid H3 fallback: if incorrect → try Mode B re-parse
            # ===========================================================
            if result and result.get("correct") is False:
                reason = (result.get("reason") or "").lower()
                retry_reasons = [
                    "wrong sum",
                    "wrong sum of numbers",
                    "secret mismatch",
                    "bad answer format",
                    "invalid json",
                    "incorrect"
                ]

                if any(r in reason for r in retry_reasons):
                    logger.warning(f"[FALLBACK] Strict Mode B re-parse triggered. reason={reason}")

                    task_info_strict = self.parse_quiz_content_strict(quiz_text)
                    if task_info_strict:
                        answer2 = self.solve_task(task_info_strict, resolved_files)
                        if isinstance(answer2, str):
                            answer2 = self._strip_code_fences(answer2)

                        logger.info(f"[FALLBACK] Resubmitting corrected answer: {answer2}")

                        result2 = self.submit_answer(
                            task_info_strict.get("submit_url"),
                            email,
                            secret,
                            str(quiz_url),
                            answer2
                        )
                        return result2 or result

            return result

        except Exception as e:
            logger.error(f"Error solving quiz: {e}")
            logger.error(traceback.format_exc())
            return None

    # ===========================================================
    # Mode A (light parse)
    # ===========================================================
    def parse_quiz_content(self, content: Any):
        content_str = str(content)

        prompt = f"""
Extract the following from the quiz content:

1. Main task/question
2. Submit URL
3. File URLs (list)
4. Expected answer format (string|number|json|boolean|base64)

Quiz Content:
{content_str}

Return ONLY valid JSON:
{{
  "task": "...",
  "submit_url": "...",
  "file_urls": [],
  "answer_format": "string"
}}
"""

        resp = self.llm.get_completion(prompt, mode="A")
        if not resp:
            return None

        return self._extract_first_json(resp)

    # ===========================================================
    # Mode B (strict fallback)
    # ===========================================================
    def parse_quiz_content_strict(self, content: Any):
        content_str = str(content)

        prompt = f"""
You MUST extract a correct JSON specification from the quiz content.

Quiz content:
{content_str}

Return ONLY JSON in this structure:

{{
  "task": "...",
  "submit_url": "...",
  "file_urls": [],
  "answer_format": "string|number|json|boolean|base64"
}}
"""

        resp = self.llm.get_completion(prompt, mode="B")
        return self._extract_first_json(resp)

    # ===========================================================
    # Task solver
    # ===========================================================
    def solve_task(self, task_info: dict, resolved_files: list):
        try:
            task_text = str(task_info.get("task", "") or "")
            answer_format = task_info.get("answer_format", "string")

            # Secret-based tasks
            for rf in resolved_files:
                if rf.get("type") == "secret":
                    val = rf.get("value")
                    if answer_format == "number":
                        try:
                            return int(val)
                        except Exception:
                            return float(val)
                    if answer_format == "json":
                        try:
                            return json.loads(val)
                        except Exception:
                            return val
                    return val

            # CSV tasks
            csv_entries = [r for r in resolved_files if r.get("type") == "csv"]
            if csv_entries:
                csv_url = csv_entries[0]["url"]

                cutoff = None
                m = re.search(r"Cutoff[:\s]+(\d+)", task_text, re.IGNORECASE)
                if m:
                    cutoff = int(m.group(1))

                try:
                    result = self._compute_sum_from_csv(csv_url, cutoff=cutoff)
                    if answer_format == "number":
                        return result
                    if answer_format == "json":
                        return {"result": result}
                    return str(result)
                except Exception as e:
                    logger.error(f"CSV error: {e}")

            # Handle downloaded resources via DataProcessor
            processed_data = {}
            for rf in resolved_files:
                try:
                    rtype = rf.get("type")
                    url = rf.get("url")

                    if rtype == "html":
                        processed_data[rf.get("source")] = rf.get("html")[:2000]

                    elif rtype in ["download", "pdf", "audio"]:
                        path = self.data_processor.download_file(url)
                        if path:
                            processed_data[url] = self.data_processor.process_file(path)

                except Exception as e:
                    logger.error(f"Error processing file {rf}: {e}")

            # If we have auxiliary data → ask LLM for structured plan
            if processed_data:
                plan_prompt = f"""
I have this task: {task_text}

Processed data previews:
{json.dumps(processed_data, default=str)}

Return ONLY JSON:
{{
  "action": "...",
  "value_column": "...",
  "filter": "...",
  "format": "number|string|json|boolean"
}}
"""
                plan_resp = self.llm.get_completion(plan_prompt, mode="A")
                plan_json = self._extract_first_json(plan_resp)

                if plan_json and csv_entries:
                    try:
                        action = plan_json.get("action", "").lower()
                        val_col = plan_json.get("value_column")
                        filter_expr = plan_json.get("filter")

                        if action.startswith("sum"):
                            df = pd.read_csv(csv_entries[0]["url"])
                            if filter_expr:
                                m = re.search(r"(\w+)\s*[<>=]+\s*(\d+)", filter_expr)
                                if m:
                                    col, num = m.group(1), float(m.group(2))
                                    df = df[df[col].astype(float) <= num]

                            if val_col and val_col in df.columns:
                                s = df[val_col].astype(float).sum()
                                return s
                            else:
                                return df.select_dtypes(include="number").sum().sum()

                    except Exception as e:
                        logger.error(f"Plan exec error: {e}")

                # Worst case: let LLM compute answer
                fallback_prompt = f"""
Using the processed data below, solve the task:

Task: {task_text}
Required answer format: {answer_format}

Processed data:
{json.dumps(processed_data, default=str)}

Return ONLY the answer.
"""
                fallback_resp = self.llm.get_completion(fallback_prompt, mode="A")
                fallback_resp = self._strip_code_fences(fallback_resp)

                if answer_format == "number":
                    return float(fallback_resp) if "." in fallback_resp else int(fallback_resp)
                if answer_format == "json":
                    return json.loads(fallback_resp)
                if answer_format == "boolean":
                    return fallback_resp.strip().lower() in ("true", "yes", "1")
                return fallback_resp

            # Nothing detected
            logger.error("No deterministic path found")
            return None

        except Exception as e:
            logger.error(f"Task solve error: {e}")
            logger.error(traceback.format_exc())
            return None

    # ===========================================================
    # Submit answer
    # ===========================================================
    def submit_answer(self, submit_url: Any, email: str, secret: str, quiz_url: str, answer: Any):
        """Submit the answer to the specified endpoint"""
        try:
            # Normalize submit_url to absolute
            submit_url_str = self._normalize_url(quiz_url, submit_url)
            if not submit_url_str:
                logger.error(f"Invalid submit_url: {submit_url}")
                return None

            # Ensure answer is serialized to a safe type (avoid sending JSON object)
            serialized_answer = self._serialize_answer(answer)

            payload = {
                'email': email,
                'secret': secret,
                'url': quiz_url,
                'answer': serialized_answer
            }

            logger.info(f'Submitting to: {submit_url_str}')
            try:
                logger.info(f'Payload: {json.dumps(payload, indent=2)}')
            except Exception:
                # fallback logging to avoid serialization errors
                logger.info(f'Payload (unserializable) - email={email} url={quiz_url} answer_type={type(serialized_answer)}')

            response = requests.post(
                submit_url_str,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            logger.info(f'Response status: {response.status_code}')

            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    logger.error("Failed to decode JSON response from submit")
                    return None
                logger.info(f'Response JSON: {json.dumps(result, indent=2)}')
                return result
            else:
                # log response body for debugging
                text = response.text[:2000] if response.text else "<no body>"
                logger.error(f'Failed to submit: status={response.status_code} body={text}')
                return None

        except Exception as e:
            logger.error(f'Error submitting answer: {str(e)}')
            logger.error(traceback.format_exc())
            return None

