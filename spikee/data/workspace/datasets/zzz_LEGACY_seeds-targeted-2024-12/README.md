# Dataset generation

The dataset can be generated using this command:

```bash
$ spikee generate --include-standalone-inputs --seed-folder datasets/seeds-targeted-2024-12 --format full-prompt

Dataset generated and saved to datasets/seeds-targeted-2024-12.jsonl

=== Dataset Statistics ===
Total Entries: 672

Breakdown by Jailbreak Type:
Jailbreak Type      Count
----------------  -------
new-instructions       60
sorry                  16
dan                    92
ignore                 76
test                   92
errors                 32
debug                  32
dev                    60
emergency              16
no-limits              32
no-constraints         16
experimental           32
hidden-function        16
academic               16
new-task               48
challenge              16
training               16
None                    4

Breakdown by Instruction Type:
Instruction Type       Count
-------------------  -------
data-exfil-markdown      192
xss                      172
encoding                 152
translation               76
long-output               76
None                       4

Breakdown by Language:
Language      Count
----------  -------
en              612
zu               30
gd               30

Breakdown by Task Type:
Task Type        Count
-------------  -------
summarization      334
qna                334
None                 4

Breakdown by Suffix ID:
Suffix ID      Count
-----------  -------
None             672
```

## Dataset with splotlighting 

You can also include spotlighting and adversarial suffixes to see if that has an impact on the attack success rate:

```bash
$ spikee generate --include-standalone-inputs --seed-folder datasets/seeds-targeted-2024-12 --spotlighting-data-markers $'\nDOCUMENT\n',$'\n<data>\nDOCUMENT\n</data>\n' --format full-prompt

W/SPIKE - Simple Prompt Injection Kit for Exploitation
Author: Reversec

Dataset generated and saved to datasets/seeds-targeted-2024-12.jsonl

=== Dataset Statistics ===
Total Entries: 3820

Breakdown by Jailbreak Type:
Jailbreak Type      Count
----------------  -------
new-instructions      360
sorry                  96
dan                   456
ignore                360
test                  552
errors                192
debug                 192
dev                   360
emergency              96
no-limits             192
no-constraints         96
experimental          192
hidden-function        96
academic               96
new-task              288
challenge              96
training               96
None                    4

Breakdown by Instruction Type:
Instruction Type       Count
-------------------  -------
data-exfil-markdown     1104
xss                      984
encoding                 864
translation              432
long-output              432
None                       4

Breakdown by Language:
Language      Count
----------  -------
en             3460
zu              180
gd              180

Breakdown by Task Type:
Task Type        Count
-------------  -------
summarization     1908
qna               1908
None                 4

Breakdown by Suffix ID:
Suffix ID               Count
--------------------  -------
None                     1276
adv-suffix-random-01     1272
adv-suffix-random-03     1272
```

