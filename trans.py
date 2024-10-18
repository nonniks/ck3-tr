

from dataclasses import dataclass, field
import json
import os
import argparse
import logging
import asyncio
import re
from typing import Optional
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)

try:
    with open("config.json", "r") as file:
        config = json.load(file)
except FileNotFoundError:
    config = {
        "mod_path": None,
        "mock_translation": False,
        "batch_size_bytes": 800,
        "cache_file": "translation_cache.json",
        "openai_key": None,
        "max_concurrent_tasks": 1
    }

def find_yml_files(base_path : str) -> list[str]:
    yml_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".yml"):
                yml_files.append(os.path.join(root, file))
    return yml_files


@dataclass
class Substitution:
    # replace src with dst in the whole
    src: str
    dst: str


MESSAGE_LINE_START_PATTERN = re.compile(r"""
    ^                  # Начало строки
    \s*                # Пробелы
    [a-zA-Z0-9_\-.]+    # идентификатор, например unify_italian_empire_decision, или pantheon_restoration.0001.t:
    (:\d*)?           # возможно за ним следует : и возможно число, например unify_italian_empire_decision:1
    \s*                # завершающие пробелы
""", re.VERBOSE)

# Регулярные выражения для поиска переменных
VARIABLE_PATTERNS = [
    re.compile(r"\[.*?\]"),  # Для переменных вида [...]
    re.compile(r"\{.*?\}"),  # Для переменных вида {...}
    re.compile(r"\$.*?\$"),  # Для переменных вида $...$
]

class FailedParse(Exception):
    pass

class Cache:
    def __init__(self, path):
        self.path = path
        self.cache = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                self.cache = json.load(file)
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, value):
        self.cache[key] = value

    def save(self):
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(self.cache, file, ensure_ascii=False, indent=2)


# Очистка переведенного текста
def clean_translation(text):
    return text.replace('```yaml', '').replace('```', '').replace('/n', '').strip()

# Кидаем исключение, если не найден перевод для данной строки.
class MissingTranslation(Exception):
    pass

def try_parse_and_translate_line(line: str, translation_dict: dict[str, str]) -> str:
    if line.strip() == 'l_english:': # Начало блока перевода
        return line.replace('english', 'russian')
    if not line.strip(): # Пустая строка
        return line
    if line.lstrip().startswith('#'): # Комментарий
        return line
    m = re.match(MESSAGE_LINE_START_PATTERN, line)
    if not m:
        raise FailedParse("MESSAGE_LINE_START_PATTERN не заматчился")
    prefix = m.group()
    quoted_part_with_quotes = line[m.end():].strip()
    if not (quoted_part_with_quotes.startswith('"') and quoted_part_with_quotes.endswith('"')):
        raise FailedParse(f"Вторая часть строки {quoted_part_with_quotes!r} не начинается и не заканчивается кавычками")
    quoted_part = quoted_part_with_quotes[1:-1]        
    substitutions = {}

    current_marker_idx = 0
    def replace_with_marker(match):
        nonlocal current_marker_idx, substitutions
        marker = f"V{current_marker_idx}"
        current_marker_idx += 1
        substitutions[marker] = match.group()
        return marker

    # Заменяем переменные в строке на V1, V2, V3, ...
    for pattern in VARIABLE_PATTERNS:
        quoted_part = pattern.sub(replace_with_marker, quoted_part)
        for match in pattern.finditer(quoted_part):
            marker = f"V{len(substitutions)}"
            quoted_part = quoted_part[:match.start()] + (' '*len(match.group())) + quoted_part[match.end():]
            substitutions[marker] = match.group()

    NL_REPLACEMENT = 'NL'
    if '\\n' in quoted_part:
        quoted_part = quoted_part.replace('\\n', NL_REPLACEMENT)

    if '\\' in quoted_part:
        raise FailedParse(f"В строке {quoted_part!r} есть символ \\ и это не \\n")

    quoted_part_without_variables = re.sub(r'V[0-9]+', '', quoted_part)
    if quoted_part_without_variables.strip():
        if quoted_part.strip() == 'the':
            quoted_part = ''
        else:
            # В строке есть ещё текст кроме переменных
            if translation_dict.get(quoted_part) is None:
                raise MissingTranslation(quoted_part)
            else:
                quoted_part = translation_dict[quoted_part].strip()      

    quoted_part = re.sub(r'V[0-9]+', (lambda m: substitutions[m.group()]), quoted_part)

    quoted_part = quoted_part.replace(NL_REPLACEMENT, '\\n')

    return f'{prefix}"{quoted_part}"\n'

def parse_and_translate_line_or_die(line: str, translation_dict: dict[str, str], ignore_missing_translation: bool) -> str | None:
    try:
        return try_parse_and_translate_line(line, translation_dict)
    except MissingTranslation as e:
        if not ignore_missing_translation:
            logging.error(f"Баг: Не найден перевод для строки: '{line}'")
            raise SystemExit(1)
        translation_dict[e.args[0]] = None
        return None
    except FailedParse as e:
        logging.error(f"Не удалось распарсить строку: '{line}', причина: {e}")
        raise SystemExit(1)

def batch_texts(texts: list[str], batch_size_bytes) -> list[str]:
    batches = []
    cur_batch = []
    cur_batch_size = 0

    def flush():
        nonlocal cur_batch, cur_batch_size, batches

        if not cur_batch:
            return
        batches.append(list(cur_batch))
        cur_batch.clear()
        cur_batch_size = 0

    def add(text):
        nonlocal cur_batch, cur_batch_size, batch_size_bytes, flush

        cur_batch.append(text)
        cur_batch_size += len(text.encode('utf-8'))
        if cur_batch_size > batch_size_bytes:
            flush()
            cur_batch_size = 0

    for text in texts:
        if '\n' in text:
            raise ValueError("Texts should not contain newlines")
        add(text)
    flush()
    return batches

async def translate_text(text: str, openai_client: Optional[AsyncOpenAI]) -> str:
    if openai_client is None:
        return '\n'.join('openai_translation(' + line + ')' for line in text.split('\n'))
    # Calling openai API here...

    # Инструкция для модели
    instruction = (
        "Ты переводишь YAML файл для игры Crusader Kings 3, историческую стратегическую игру, "
        "которая происходит в средневековье. Тексты могут содержать имена, фамилии, титулы, названия династий, "
        "названия культур и титулов, а также приставки к именам. Перевод должен быть выполнен на **полный русский язык**, "
        "исключив любые оригинальные английские слова или дублирования. Например, если слово 'Xungul' переводится как 'Сунгул', "
        "нельзя оставлять оригинал, должно быть только 'Сунгул'."
        "Некоторые названия или термины могут содержать символы с диакритическими знаками, их нужно упрощать до русских форм. "
        "Ключи и переменные в тексте заменены маркерами NL, K1, V1 и т.д., их не нужно переводить. Переводи только текст, оставив маркеры на своих местах. "
        "Количество строк на выходе должно ОБЯЗАТЕЛЬНО быть равно количеству строк во входе. "
        "НЕ ВСТАВЛЯЙ ЛИШНИЕ ПЕРЕВОДЫ СТРОК!!! "
    )

    response = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Переведи строки с английского на русский язык:\n{text}"}
        ],
        model="gpt-4o-mini",
        max_tokens=1 + int(len(text) * 0.8)
    )
    return clean_translation(response.choices[0].message.content.strip())


async def translate_batch_cached(batch: list[str], openai_client: Optional[AsyncOpenAI], cache: Cache, semaphore: asyncio.Semaphore) -> list[str]:
    async with semaphore:
        result = {}
        missing_from_cache = []
        for text in batch:
            cached = cache.get(text)
            if cached is not None:
                result[text] = cached
            else:
                missing_from_cache.append(text)
        if missing_from_cache:
            translated_missing = await translate_batch(batch, openai_client)
            for text, translated in zip(missing_from_cache, translated_missing):
                result[text] = translated
                cache.put(text, translated)
        result_list = []
        for text in batch:
            assert text in result
            result_list.append(result[text])
        return result_list


async def translate_batch(batch: list[str], openai_client: Optional[AsyncOpenAI]) -> list[str]:
    batch_text = '\n'.join(batch)
    logging.info("Translating batch of %d lines and %d bytes", len(batch), len(batch_text.encode('utf-8')))
    translated_batch_text = await translate_text(batch_text, openai_client=openai_client)
    translated_batch = [line.strip() for line in translated_batch_text.split('\n') if line.strip()]
    if len(batch) != len(translated_batch):
        logging.error("Original batch text:\n%s\n", batch_text)
        logging.error("Translated batch text:\n%s\n", translated_batch_text)
        logging.error("Translated '''%s''' into '''%s''', unexpected number of translations: %d != %d.", batch_text, translated_batch_text, len(batch), len(translated_batch))
        raise ValueError("Unexpected number of translations: %d != %d" % (len(batch), len(translated_batch)))
    return translated_batch

async def translate(texts: list[str], openai_client: Optional[AsyncOpenAI], cache: Cache) -> list[str]:
    batches = batch_texts(texts, config['batch_size_bytes'])
    # XXX: 1
    semaphore = asyncio.Semaphore(config['max_concurrent_tasks'])
    tasks = [asyncio.create_task(translate_batch_cached(batch, openai_client=openai_client, cache=cache, semaphore=semaphore)) for batch in batches]
    result = []
    for task in tasks:
        await task
        result.extend(task.result())
    return result

async def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mod_path', type=str, help='Путь к папке мода для перевода')
    argparser.add_argument('--mock-translation', action='store_true', help='Использовать заглушку вместо OpenAI API')
    argparser.add_argument('--batch-size-bytes', type=int, help='Максимальный размер батча в байтах')
    argparser.add_argument('--cache-file', type=str, help='Файл кэша переводов')
    argparser.add_argument('--openai-key', type=str, help='API ключ OpenAI')
    argparser.add_argument('--max_concurrent-tasks', type=int, help='Максимальное количество одновременных задач')
    args = argparser.parse_args()

    if args.mod_path:
        config['mod_path'] = args.mod_path
    if args.mock_translation:
        config['mock_translation'] = True
    if args.batch_size_bytes:
        config['batch_size_bytes'] = args.batch_size_bytes
    if args.cache_file:
        config['cache_file'] = args.cache_file
    if args.openai_key:
        config['openai_key'] = args.openai_key
    if args.max_concurrent_tasks:
        config['max_concurrent_tasks'] = args.max_concurrent_tasks

    mod_path = config['mod_path'] or input("Введите путь к папке мода для перевода: ")

    if config['mock_translation']:
        if config['cache_file'] is not None:
            config['cache_file'] += '.mock.json'

    if not mod_path:
        logging.error("Не указан путь к папке мода для перевода")
        return
    
    openai_client = None
    if not config['mock_translation']:
        openai_client = AsyncOpenAI(api_key=config['openai_key'])

    yml_files = find_yml_files(config['mod_path'])

    translation_dict = {}
    for yml_file in yml_files:
        with open(yml_file, "r", encoding="utf-8-sig") as file:
            for l in file.readlines():
                parse_and_translate_line_or_die(l, translation_dict, ignore_missing_translation=True)

    # Переводим
    src_texts = []
    for src_text in translation_dict:
        dst_text = translation_dict[src_text]
        if dst_text is None:
            src_texts.append(src_text)
    logging.info("Translating %d texts", len(src_texts))
    cache = Cache(config['cache_file'])
    try:
        dst_texts = await translate(src_texts, openai_client=openai_client, cache=cache)
    finally:
        cache.save()
    for src_text, dst_text in zip(src_texts, dst_texts):
        translation_dict[src_text] = dst_text
            
    logging.info("Writing translations")
    # Ещё раз проходим по файлам и уже заменяем
    for yml_file in yml_files:
        with open(yml_file, "r", encoding="utf-8-sig") as file:
            translated_lines = []
            for l in file.readlines():
                translated_lines.append(
                    parse_and_translate_line_or_die(l, translation_dict, ignore_missing_translation=False)
                )
        dst_file = yml_file.replace("english", "russian")
        assert dst_file != yml_file, dst_file
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with open(dst_file, "w", encoding="utf-8-sig") as file:
            file.writelines(translated_lines)
    

if __name__ == "__main__":
    asyncio.run(main())