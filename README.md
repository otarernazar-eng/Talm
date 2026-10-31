# Soup Kernel Challenge: Решение команды Bitston (Decentrathon 6.0)

Кастомные Triton-кернелы и оптимизированные autograd-примитивы для ускорения обучения `soup train` (`soup-cli==0.77.0`, модель `Qwen/Qwen2.5-1.5B-Instruct`, веса NF4 4-bit, LoRA $r=16, \alpha=32$ на `q_proj` и `v_proj`, batch 4, seq 512, 64 шага) на одном GPU **Tesla T4 (sm_75, FP16)** в Kaggle.

- **Репозиторий команды**: [BAITC-Hacks/hack-6e10e900-bitston](https://github.com/BAITC-Hacks/hack-6e10e900-bitston)
- **Точка входа**: `solution.apply()` в `solution/__init__.py`.

---

## 1. Результаты бенчмарка (Kaggle Tesla T4, формат жюри `--repeats 2`)

Замер производился официальным раннером `harness/run_task.py --repeats 2` (чередование: `база -> решение -> решение -> база`) в изолированной среде Kaggle T4.

| Метрика | Базовый Soup 0.77.0 | Решение команды Bitston | Отношение / Выигрыш |
|---|---|---|---|
| **Время `soup train` (wall time)** | **85.0 с** | **37.7 с** | **Speedup 2.251×** |
| **Время цикла обучения (64 шага)** | 56.5 с | 24.1 с | **2.34× быстрее** |
| **Пиковая память GPU (nvidia-smi)** | **14 345 MiB** | **3 461 MiB** | **Снижение памяти 4.145×** |
| **PyTorch Max Allocated** | 13 373 MiB | 3 214 MiB | 4.16× меньше |
| **Evaluation Loss (public eval)** | 2.04968 | 2.04983 | **+0.007%** (порог $\le 0.5\%$) |
| **Доля своих Triton-кернелов** | 0.0% | **12.4%** | **Выполнено** (порог $\ge 5.0\%$) |
| **Проверки правил (check_rules.py)** | — | **10 / 10 PASS** | Все 10 проверок пройдены |

---

## 2. Ключевые технические оптимизации

1. **Fused Vocab Chunked Head (`solution/fused_head.py`)**:
   * *Проблема базы:* Создание и хранение в памяти огромного тензора логитов $[4, 512, 151936]$ в FP16 и 3 его FP32-копий в backward ($>3.6$ ГБ VRAM).
   * *Решение:* Проход по словарю срезами по $8192$ столбца сразу для всех токенов батча. Онлайн-расчет log-sum-exp, локальный argmax и аккумуляция градиента $dH$ на месте. Тензор логитов вообще не создается.
   * *Эффект:* Снижение пиковой памяти более чем на 4 ГБ.
2. **Grouped Layer NF4 Dequantization (`solution/qgroup.py`)**:
   * *Проблема базы:* `bitsandbytes` вызывает деквантование для каждой проекции отдельно (389 мелких вызовов за шаг, создающих очереди ожидания CPU-GPU).
   * *Решение:* 1 вызов кастомного Triton-кернела распаковывает сразу все 7 матриц 4-битного слоя за один проход на максимальной скорости шины T4 ($\approx 200$ ГБ/с).
3. **Padding Removal & Varlen Attention (`solution/unpad.py`)**:
   * *Проблема базы:* Около 45% токенов в батче — это бесполезный pad.
   * *Решение:* Упаковка токенов батча в плоский вектор ($1 \times N$) с каузальным вниманием через `cu_seqlens` (memory-efficient attention) и stride-0 GQA трансляцией KV-голов.
4. **Fused RMSNorm + Residual Add (`solution/rmsnorm.py`)**:
   * *Решение:* Слияние сложения residual-связи и нормализации в один кернел с прямым выводом в FP16.
5. **Fused RoPE (`solution/rope.py`)**:
   * *Решение:* Слитный поворот $Q$ и $K$ в одном кернеле без выделения промежуточных тензоров.
6. **Host-Side Bookkeeping & Deferred Metrics (`solution/host.py`)**:
   * *Решение:* Устранение блокирующих синхронизаций GPU между forward и backward. Метрики TRL и запись строк в `rewind.jsonl` откладываются до шага логирования.

---

## 3. Пошаговые команды для запуска в Kaggle Notebook (GPU T4 x1)

Создайте новый Kaggle Notebook с ускорителем **GPU T4 x1** и вставьте команды в ячейки по порядку:

### Ячейка 1: Установка зависимостей с точными версиями
```python
!pip install -q \
    "soup-cli[train]==0.77.0" \
    "transformers==5.16.1" \
    "peft==0.20.0" \
    "trl==0.29.1" \
    "bitsandbytes==0.50.2" \
    "accelerate==1.14.0" \
    pytest
```

### Ячейка 2: Клонирование репозитория команды Bitston
```python
import os, shutil

# Очищаем рабочую директорию от старого клона (если перезапускается ячейка)
if os.path.exists("/kaggle/working/repo"):
    shutil.rmtree("/kaggle/working/repo")

!git clone https://github.com/BAITC-Hacks/hack-6e10e900-bitston.git /kaggle/working/repo
%cd /kaggle/working/repo
!git log -1 --oneline
```

### Ячейка 3: Подготовка данных
```python
%cd /kaggle/working/repo
!python task/prepare_data.py
```

### Ячейка 4: Прогон тестов решения
```python
%cd /kaggle/working/repo
!python -m pytest solution -q
```

### Ячейка 5: Запуск официального бенчмарка (harness)
```python
%cd /kaggle/working/repo

# Формат регламента жюри (2 повтора: база -> решение -> решение -> база)
!python harness/run_task.py --repeats 2
```

### Ячейка 6: Проверка 10 правил регламента и подсчет очков
```python
%cd /kaggle/working/repo

# 1. Проверка всех 10 правил регламента (должно быть ALL CHECKS PASSED)
!python harness/check_rules.py

# 2. Итоговый вывод ускорения (speedup) и сжатия памяти (memory ratio)
!python harness/score.py runs/result.json
```

---

## 4. Соблюдение правил регламента (10/10 PASS)

```
PASS  1. every base and solution run finished                                            2 base and 2 solution runs of 2 each
PASS  2. no ready-made kernel libraries                                                  loaded: none; imported in solution/: none
PASS  3. own @triton.jit kernels in solution/ run in training                            fused_head_vocab_kernel, rmsnorm_kernel, rope_kernel, nf4_group_dequant_kernel
PASS  4. own kernels together take >= 5% of the training time                            12.4% of the training time (threshold 5%)
PASS  5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind)  identical within tolerance
PASS  6. loss curve follows the base                                                     first step 0.03%, mean of all steps 0.05% (max 1% and 3%)
PASS  7. quality: eval loss at most 0.5% above the base                                  +0.007%
PASS  8. not worse than the base on time or memory (10% margin)                          time x0.444, memory x0.241 of the base
PASS  9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree)    clean
PASS  10. Soup regression tests with your patches                                        79 passed in 0.58s

ALL CHECKS PASSED
speedup 2.251x, memory base/solution 4.145x, eval loss vs base +0.007%
```
