# Hand answers for the 19 validation conversations (agent, from the turbo transcripts)

Source: `request_dump/transcripts/*.large-v3-turbo.json`, read in full on 2026-09-17.
One table per conversation. Columns: question number, my answer, transcript segment(s) the
proof sits in (`#nn` as printed by `bench/mine/dump_val.py`), my best guess at the annotated
span in seconds (segment bounds, trimmed by word count where the fact is a sub-sentence), the
quote, and a note. `model:` marks where the served pipeline (run F) answered differently.
Rows flagged **CHECK** are the ones worth listening to by ear.

This file is the source of truth: edit an answer or a time here and re-run
`python bench/mine/answers_md.py` to rebuild `bench/mine/agent_labels/*.json` and the probe table.

Yes count: 95 of 190 (the training set is 195/195). Nine answers differ from the served model (run F); `python bench/mine/answers_md.py --diff` lists them.

## conversation_sample_3

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #16 | 67.3 | 72.44 | "The influenza one I am giving you now. That is Influvac Tetra." Given at this visit; the pneumococcal one is only prescribed. model: no. **CHECK**: "already" could be read as "before the visit"; I read it as the note's "InfluvacTetra given" against "prescription for Prevenar/Pneumovax". |
| 2 | yes | #10-#12 | 46.64 | 49.38 | "I want to check there are no acute signs of illness today … And are there? None. You are clear to be vaccinated now." Seed is #12; the full exchange is 38.08-49.38. |
| 3 | yes | #05-#07 | 26.56 | 32.44 | "First, the side effects. … I have gone through what you can expect after the injection, so you know what is normal and what is not." |
| 4 | no | | | | Second vaccine asked about was pneumococcal (#13), not a tetanus booster. |
| 5 | yes | #03 | 13.94 | 19.28 | "I would like the influenza vaccination. I am hoping to get it done before the season really starts." |
| 6 | no | | | | Off-topic; bus and holiday small talk only (#20-#23). |
| 7 | yes | #18 | 74.10 | 79.2 | "I am creating a prescription for Previnar 13 and Pneumovax 23." |
| 8 | no | | | | Off-topic. |
| 9 | no | | | | Opposite: "None. You are clear to be vaccinated now." (#12). |
| 10 | yes | #14 | 58.0 | 62.18 | "So today is a vaccination request and a risk assessment together." |

## conversation_sample_7

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #06 | 27.78 | 31.34 | "Mostly the right side. That is where it settles." |
| 2 | yes | #44 | 154.52 | 158.30 | "Then I want you to continue the pancillin for a further five days." |
| 3 | no | | | | Opposite: "You have improved since the antibiotic treatment." (#40). |
| 4 | no | | | | Opposite: "I also note you are slightly overweight." (#30). |
| 5 | no | | | | Five days, not ten (#44-#45). |
| 6 | yes | #08 | 35.40 | 41.02 | "A lot of wind. An embarrassing amount, honestly." Alternative: the finding "Your abdomen is distended with gas." (#20, 85.84-88.50). |
| 7 | no | | | | Opposite: "Normal. Completely normal." (#10). |
| 8 | no | | | | Twin of q1; right side. |
| 9 | no | | | | Off-topic. |
| 10 | no | | | | Off-topic. |

## conversation_sample_8

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #10 | 71.58 | 75.8 | "There is redness inside your mouth, and there are coatings." |
| 2 | yes | #06-#08 | 43.6 | 56.3 | "Yes. Tablets called Fluconazole once, and a gel … And did either of those help? No, neither of them made any difference." Long; the tight alternative is the "No, neither …" answer (53.6-56.3). |
| 3 | yes | #07 | 46.52 | 50.5 | "and a gel. Brenton, I think it was called." model: no. Brentan was tried (and did not help). |
| 4 | no | | | | Fungal infection; Fluconazole, no antibiotic. |
| 5 | yes | #12-#13 | 90.3 | 97.16 | "Since it has kept coming back after the earlier treatments, I would call this a persistent fungal infection of the mouth." |
| 6 | yes | #04-#05 | 29.4 | 35.9 | "It is my mouth. I keep getting that white stuff in there, and it keeps coming back." model: no. **CHECK**: I read "tendency to develop oral thrush" as the note's "recurrent oral candidiasis"; the twin negative is q7. |
| 7 | no | | | | Opposite: Brentan made no difference (#07-#08). |
| 8 | yes | #16 | 111.1 | 116.02 | "I will prescribe Fluconazole 50 milligrams once a day for seven days." |
| 9 | no | | | | Off-topic (plants, bicycle). |
| 10 | no | | | | 50 mg, not 150 mg (#16). |

## conversation_sample_11

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #04 | 25.66 | 31.30 | "Well, I am allergic to wasps, and I keep worrying about what happens if I get stung." |
| 2 | yes | #04 | 25.66 | 31.30 | Same sentence; also "With a wasp allergy, there is a risk of a severe reaction" (#09, 48.68-54.16). |
| 3 | no | | | | Off-topic (roadworks, dark evenings). |
| 4 | no | | | | The pen is for anaphylaxis after a wasp sting, not asthma (#09-#12). |
| 5 | no | | | | Opposite: "risk of a severe reaction across the whole body" (#09). |
| 6 | no | | | | 0.3 mg, not 0.15 mg (#14). |
| 7 | no | | | | Wasps, not pollen. |
| 8 | no | | | | Off-topic. |
| 9 | yes | #14 | 73.14 | 76.6 | "I am prescribing an EpiPen of 0.3 milligrams." |
| 10 | no | | | | Wasps, not bees. |

## conversation_sample_14

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #21 | 86.16 | 89.08 | "It is an allergic irritation of the eyes." |
| 2 | yes | #14 | 59.1 | 62.14 | "The white part of both eyes is red." then "It is the lining of the eye reacting … It is called the conjunctiva." (#16-#18). model: no. The finding is conjunctival redness in both eyes. |
| 3 | yes | #31 | 128.22 | 131.24 | "One drop in each eye, two times a day." |
| 4 | yes | #07 | 28.78 | 33.52 | "My eyes. I was hoping I could get some allergy drops for them." |
| 5 | yes | #09 | 37.78 | 44.80 | "The heat, mostly. And all this wind we have been having. My eyes really do not like it." Doctor's echo "So heat and windy weather set it off." is #10 (45.44-48.32). |
| 6 | no | | | | Each eye (#31). |
| 7 | no | | | | Drops asked for (#07), no tablets mentioned. |
| 8 | yes | #31 | 128.22 | 131.24 | "two times a day"; repeated in #32. |
| 9 | no | | | | "one milligram per milliliter" (#29), not 10 mg/ml. |
| 10 | no | | | | Off-topic. |

## conversation_sample_16

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | Wasps, not bees (#13-#14). |
| 2 | no | | | | Off-topic; no reflex hammer, lung measurements only. |
| 3 | no | | | | Opposite: "I smoke, and I am not motivated to stop." (#14). |
| 4 | yes | #13-#14 | 63.74 | 69.1 | "And I understand you are allergic to wasps. / Yes, wasps." |
| 5 | no | | | | Off-topic. |
| 6 | no | | | | Transcript says 423.6 L/min (#17). **CHECK** by ear: 423.6 vs 523.6. |
| 7 | yes | #10 | 44.74 | 50.64 | "Not the Spiriva. It is the cost, honestly. I have not been taking it regularly." |
| 8 | yes | #25 | 127.2 | 130.86 | "I do recommend that you take the Spiriva regularly." |
| 9 | no | | | | Transcript says "The ratio to FVC, 68%" (#17). **CHECK** by ear: 68 vs 88. |
| 10 | yes | #17 | 81.62 | 84.3 | "Your FEV1 is 82%." |

## conversation_sample_24

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | The lump is watched, not removed (#23-#25). |
| 2 | yes | #08 | 46.2 | 49.14 | "I can feel a lump on my left forearm." |
| 3 | no | | | | No antibiotics mentioned. |
| 4 | yes | #25 | 116.60 | 123.20 | "If it is unchanged at that point, I will refer you for an ultrasound scan so we can see what it is made of." |
| 5 | no | | | | Six weeks, not six months (#23). |
| 6 | yes | #17-#18 | 86.2 | 93.94 | "It could be a fatty lump, what we call a lipoma, or it could be a collection of blood under the skin, a hematoma." |
| 7 | no | | | | Left forearm. |
| 8 | no | | | | Opposite: "a possible benign change, meaning not a dangerous one." (#20-#21). |
| 9 | no | | | | Off-topic. |
| 10 | yes | #05-#06 | 27.8 | 35.9 | "Now, you are booked in today for removal of a skin change under your breast. Is that right? That is right." model: no. |

## conversation_sample_26

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #04 | 32.3 | 37.14 | "That is a benign growth, Niels. Benign meaning harmless." Also #06 (46.24-51.84) and #17 (129.10-136.88). |
| 2 | yes | #04 | 29.60 | 32.3 | "It is a seborrheic keratosis." Also "The conclusion is that it is a seborrheic keratosis, and that it is benign." (#06, 46.24-51.84). |
| 3 | yes | #01 | 11.0 | 16.24 | "Is there an answer on that skin change you took off under my breast?" |
| 4 | no | | | | Off-topic. |
| 5 | no | | | | Opposite: it was taken off before; this is the result. |
| 6 | yes | #01 | 11.0 | 16.24 | "that skin change you took off under my breast" and "when someone cuts something off you and sends it away" (#08, 56.66-65.64). model: no. The twin negative is q5. |
| 7 | no | | | | Off-topic (house, football). |
| 8 | yes | #04 | 32.3 | 37.14 | "Benign meaning harmless." Reassurance repeated at #06-#07, #13, #17. |
| 9 | yes | #11 | 81.4 | 89.70 | "What I have done is send you the information about the benign result as a written reply in the electronic consultation system." |
| 10 | no | | | | Opposite: "Benign means not malignant, not cancerous." (#17). |

## conversation_sample_30

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | Off-topic (roadworks). |
| 2 | no | | | | Cough and sore throat (#04), no earache. |
| 3 | yes | #18 | 93.5 | 97.38 | "What I recommend is supportive treatment, meaning plenty of fluids and rest." |
| 4 | no | | | | "no antibiotics … so no" (#17-#18). |
| 5 | no | | | | "No. No fever at all" (#06). |
| 6 | yes | #09 | 43.98 | 48.40 | "You have no fever now either, and I find no signs of a bacterial infection." |
| 7 | yes | #09 | 43.98 | 46.0 | "You have no fever now either"; also "your temperature being normal" (#11, 54.10-60.24). |
| 8 | no | | | | Nothing abnormal reported; findings point away from bacterial cause (#11-#12). |
| 9 | no | | | | Off-topic. |
| 10 | yes | #18 | 93.5 | 97.38 | "What I recommend is supportive treatment, meaning plenty of fluids and rest." Same span as q3. |

## conversation_sample_31

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | Antihistamine (#33), no nasal steroid. |
| 2 | no | | | | Allergic rhinitis (#27). |
| 3 | yes | #15-#16 | 46.42 | 49.60 | "Have you had a fever with it at all? No fever." Also "You have no fever" at the examination (#23, 74.92-80.02). |
| 4 | no | | | | Sneezing and blocked nose (#11). |
| 5 | no | | | | Opposite: "I have a known allergy. That has been established for a long time." (#18). |
| 6 | yes | #33 | 125.6 | 128.6 | "I am prescribing an antihistamine for you." |
| 7 | no | | | | "No fever." (#16, #23). |
| 8 | yes | #18 | 55.40 | 59.64 | "I have a known allergy. That has been established for a long time." |
| 9 | no | | | | "clear rather than colored" (#23). |
| 10 | no | | | | Off-topic. |

## conversation_sample_40

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | Pivmecillinam (#18), not nitrofurantoin. |
| 2 | yes | #10 | 44.56 | 48.80 | "It is also positive for nitrite, which points in the same direction." |
| 3 | yes | #03 | 11.12 | 15.84 | "It burns when I pass water. It has been really uncomfortable." |
| 4 | no | | | | Burning, not itching. |
| 5 | no | | | | Antibiotic prescribed (#18); no painkillers mentioned. |
| 6 | yes | #08 | 36.3 | 39.18 | "The dipstick is positive for white blood cells." |
| 7 | no | | | | "an infection in the bladder" (#16), urinary tract infection (#12). |
| 8 | no | | | | Off-topic. |
| 9 | no | | | | Positive for nitrite (#10). |
| 10 | no | | | | Positive for white blood cells (#08). |

## conversation_sample_44

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #09 | 38.26 | 44.56 | "Your hemoglobin A1c was normal, and the other tests were normal as well." |
| 2 | yes | #15 | 62.54 | 65.36 | "No treatment is needed on that basis." |
| 3 | no | | | | Off-topic. |
| 4 | no | | | | Normal (#09). |
| 5 | yes | #11 | 48.7 | 53.52 | "My assessment is that this is non-pathological fatigue." |
| 6 | yes | #09 | 38.26 | 44.56 | All results normal; "nothing is showing up in the bloods" (#10, 44.86-47.86). **CHECK**: "reassuring" is my reading of all-normal results. |
| 7 | no | | | | All normal. |
| 8 | yes | #09 | 40.7 | 44.56 | "and the other tests were normal as well." |
| 9 | yes | #10 | 44.86 | 47.86 | "All normal, so nothing is showing up in the bloods." with "Correct." (#11). |
| 10 | yes | #06 | 24.66 | 29.26 | "Fatigue, mainly. That is really the whole of it. I have simply been tired." |

## conversation_sample_45

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #02 | 9.00 | 11.4 | "I have a cough that will not shift" |
| 2 | yes | #09 | 42.22 | 44.72 | "My assessment is acute bronchitis." |
| 3 | no | | | | Acute, not chronic. |
| 4 | no | | | | Off-topic (baking). |
| 5 | yes | #02 | 11.4 | 13.48 | "and I am a little short of breath." Also "the mild breathlessness" (#11). |
| 6 | yes | #04 | 18.00 | 19.5 | "No fever at all." |
| 7 | yes | #02 | 9.00 | 13.48 | Cough and breathlessness; "It treats the airways directly, and that is where the trouble is." (#15, 65.60-69.88). **CHECK**: wording "related to the airways" is the note's category, not said verbatim. |
| 8 | yes | #05 | 24.1 | 28.00 | "I have examined you, and there are no signs of pneumonia." |
| 9 | no | | | | Inhalation therapy only; antibiotics never mentioned. |
| 10 | yes | #13 | 55.78 | 58.5 | "I would like to start inhalation therapy."; "start the inhalations" (#22). |

## conversation_sample_46

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #08 | 23.24 | 27.44 | "It is 6.2 millimoles per liter." |
| 2 | no | | | | Weekend forecast and "a long weekend coming up" are the patient's words; the doctor never asks about a holiday. |
| 3 | no | | | | "It is slightly raised." (#10). |
| 4 | yes | #03 | 8.08 | 11.52 | "That is right, and I have no complaints at all." |
| 5 | no | | | | "fasting blood glucose" (#06). |
| 6 | no | | | | Off-topic. |
| 7 | no | | | | No complaints at all (#03, #05). |
| 8 | yes | #14 | 43.96 | 47.74 | "It means we should consider pre-diabetes." |
| 9 | yes | #10 | 31.82 | 33.54 | "It is slightly raised."; "It is a mild elevation." (#12, 37.22-40.60). |
| 10 | yes | #02 | 5.32 | 7.62 | "This is your preventive health check." |

## conversation_sample_51

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #08 | 37.34 | 41.02 | "Yes. Your hemoglobin A1c is elevated." |
| 2 | no | | | | Off-topic (market, weekend rain). |
| 3 | yes | #12 | 52.88 | 55.18 | "We are starting you on metformin."; "We are beginning your treatment with it now." (#14, 60.70-64.02). |
| 4 | yes | #06 | 29.4 | 33.08 | "Your blood work confirms type 2 diabetes." |
| 5 | no | | | | Metformin, not insulin. |
| 6 | yes | #12 | 52.88 | 55.18 | "We are starting you on metformin." |
| 7 | no | | | | Confirmed (#05-#06). |
| 8 | yes | #16 | 69.06 | 72.04 | "Yes. I am referring you to the diabetes nurse." |
| 9 | yes | #06 | 28.34 | 33.08 | "It has. Your blood work confirms type 2 diabetes." |
| 10 | yes | #16 | 69.06 | 72.04 | "I am referring you to the diabetes nurse." |

## conversation_sample_60

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #05 | 18.16 | 21.52 | "Well, actually, no complaints at all." |
| 2 | yes | #16 | 47.4 | 50.76 | "The plan is that you continue your current treatment." (#17-#18 repeat it). |
| 3 | no | | | | No complaints (#05-#07). |
| 4 | no | | | | Opposite: improved (#12). |
| 5 | yes | #14 | 40.16 | 44.36 | "Taken together, your diabetes and your lipids are both well-controlled." model: no. |
| 6 | no | | | | Off-topic (supermarket). |
| 7 | yes | #12 | 35.7 | 38.10 | "And your lipids have improved as well." |
| 8 | yes | #10 | 28.60 | 32.84 | "Your long-term sugar value is 42 millimoles per mole." |
| 9 | yes | #14 | 40.16 | 44.36 | "your diabetes and your lipids are both well-controlled." |
| 10 | yes | #05 | 18.16 | 21.52 | "Well, actually, no complaints at all." in answer to "How have you been feeling?" **CHECK**: same evidence as q1. |

## conversation_sample_62

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | yes | #11 | 50.60 | 53.64 | "Your feet look normal, and your sensation is intact." |
| 2 | no | | | | Opposite: "There are no signs of nerve damage." (#13). |
| 3 | no | | | | Off-topic. |
| 4 | yes | #05 | 24.02 | 28.10 | "today is the foot check for your type 2 diabetes." |
| 5 | yes | #11 | 50.60 | 53.64 | "Your feet look normal" |
| 6 | yes | #13 | 58.8 | 61.18 | "There are no signs of nerve damage." |
| 7 | no | | | | Opposite: stable (#15). |
| 8 | yes | #15 | 66.84 | 68.58 | "Your diabetes is stable." |
| 9 | no | | | | Once a year (#17). |
| 10 | yes | #06 | 28.80 | 31.2 | "Yes. I have no complaints at all, but that is partly what worries me." model: no. |

## conversation_sample_73

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | "Any reaction? Soreness? … None at all." (#13-#14); no swelling. |
| 2 | yes | #15 | 45.00 | 47.86 | "Then the vaccination is given, with no complications." |
| 3 | no | | | | Foot and eye check reminders only (#17). |
| 4 | yes | #17 | 52.2 | 55.94 | "Your foot check and your eye check are coming up." |
| 5 | no | | | | Off-topic. |
| 6 | yes | #02 | 9.0 | 11.16 | "I am here for the influenza vaccine." |
| 7 | yes | #17 | 52.2 | 55.94 | "Your foot check and your eye check are coming up." Same span as q4. |
| 8 | no | | | | Influenza, not travel. |
| 9 | no | | | | Coming up, not done (#17-#18). |
| 10 | yes | #08 | 28.70 | 31.26 | "Well, no complaints at all." in answer to "How are you feeling in yourself today?"; also "I feel completely well." (#14, 41.86-44.56). |

## conversation_sample_80

| q | answer | segs | start | end | quote / note |
|---|---|---|---|---|---|
| 1 | no | | | | "No further treatment." (#29); no infection. |
| 2 | no | | | | Opposite: uncomplicated healing (#24-#26). |
| 3 | no | | | | Off-topic. |
| 4 | no | | | | Sutures, not a cast (#00, #06). |
| 5 | no | | | | Opposite: "There are no signs of infection." (#17). |
| 6 | yes | #00 | 0.00 | 4.76 | "I have an appointment to have my sutures removed." The patient is physically present: came in by the side door (#02-#03), sits down, sutures removed (#22). model: no. **CHECK**: I read this as the note's contact type (in person, not phone or e-consultation). Where the annotator put the span is a guess. |
| 7 | no | | | | Arthroscopy (#06), not an open hip operation. |
| 8 | yes | #08 | 51.20 | 55.90 | "Uncomplicated, thankfully. Nothing out of the ordinary at any point since the arthroscopy." Conclusion repeated at #24-#26 (111.72-124.94). |
| 9 | no | | | | Off-topic. |
| 10 | yes | #17 | 89.92 | 94.44 | "There are no signs of infection. Nothing there to suggest a problem." |
