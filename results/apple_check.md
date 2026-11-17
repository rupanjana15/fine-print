# Apple's iTunes terms: the model vs ToS;DR's reviewers (v1.1)

**Why.** The first real-browser test of the extension (D46) ran on apple.com/legal/internet-services/itunes and
gave a D with 2 blockers. This checks those flags against ToS;DR's human points for the same document.

**Caveat first: Apple was in training.** Apple Services (ToS;DR service 158, grade C) is in the train split, so
this shows how well the model reproduces labels it learned. It says nothing about unseen companies; D43 covers
those.

**Method.** The shipped 4-bit model ran on ToS;DR's stored copy of the iTunes Terms of Service (document 225,
52,875 characters, 83 chunks) on a free Kaggle CPU (`notebooks/kaggle_apple.py`; 34 min on 4 cores, against
~3.5 min on the M1). The Mac run was stopped because the laptop got hot. Output:
[apple_check.json](apple_check.json).

**Result.** 18 flags, every quote found word for word. Grade D (score −6: 2 blocker, 2 bad, 0 good distinct issues).

ToS;DR has **4 approved points on this document, all neutral**:

| ToS;DR point | Model |
|---|---|
| You indemnify the service | ✅ same point, neutral |
| Provided "as is", at your own risk | ✅ same point, neutral |
| No warranty of an uninterrupted / error-free service | ✅ same point, neutral |
| No responsibility for linked websites | ≈ same category, different point ("no liability for damages") |

The model's serious flags, against Apple's 58 points across its 5 documents:

| Model flag | Quote (short) | Verdict |
|---|---|---|
| Blocker: content can be deleted / edited without reason | "Apple may monitor and decide to remove or edit any submitted material…" | ✅ one of Apple's 2 ToS;DR blockers (recorded on the Media Services terms) |
| Bad: continued use = consent to changed terms | "Your continued use of the Services will be deemed acceptance thereof." | ✅ an Apple bad point (Media Services terms) |
| Bad: liability limited to the fees you paid | "…total liability… exceed… two hundred and fifty dollars ($250.00)" | ≈ right issue, wrong case: it's a fixed $250 cap, not "fees paid" |
| Blocker: private messages can be read | "Apple may monitor your use of the Services and Content to ensure that you are following these Usage Rules." | ❌ false alarm: monitoring use is not reading messages |

**Grade.** The model's D is one grade harsher than ToS;DR's C for Apple as a whole. Graded by the same rule,
ToS;DR's 4 points on this document alone give a C. Without the false blocker the score is −4, still a D. On this
page the model is stricter than the reviewers, the opposite of D43's average tendency. It found real issues that
ToS;DR recorded on a sister document but not on this one.

**Takeaways.**
- Quotes are reliable (18/18), and explanations follow ToS;DR's case wording, sometimes picking a neighbouring
  case ($250 cap → "fees you paid").
- Blockers are rare and costly (D45). Here 1 of 2 was wrong, and it alone moves the score by 2. A single page's
  grade is fragile, which is why the extension and the demo show the flags and their quotes, not just the letter.
- The live page in the extension gave a different second blocker ("personal information may be collected…") than
  this archived copy. Apple's current text differs from ToS;DR's snapshot.
