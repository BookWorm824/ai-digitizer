# AI Embroidery Digitizer — Build Brief

Nov 19, 2026 · @Lauren

*(Converted from "AI Embroidery Digitizer — Build Brief.pdf". The PDF is the original.)*

## What we're building

A one-click app that turns any image into a sew-ready embroidery file. The user uploads a
picture, ChatGPT redraws it as clean embroidery-friendly art, and the app digitizes it into
stitches and exports a machine file (PES, DST, JEF and others).

The promise: upload → approve the cleaned-up art → pick size and fabric → download. No
digitizing knowledge needed.

Version 1 runs on Lauren's Windows PC, in the browser, for Morgan's Corner Shop custom
orders. It's built so it could become a hosted product later without a rewrite.

## User-friendly rules (non-negotiable)

- **Four screens max** from upload to download. One job per screen.
- **Plain words only.** Terms like underlay, pull compensation and tatami never appear on
  main screens. They live in an Advanced drawer that starts closed.
- **Smart defaults.** Pressing Next on every screen must still produce a good file.
- **Every problem comes with a one-click fix.** Example: "This text is too small to sew
  cleanly." [Make it bigger] [Remove it]
- **Show, don't tell.** Before/after slider for the art; realistic thread preview for the
  stitches.
- **Nothing gets lost.** Undo on every step, autosaved projects, going back never wipes
  later choices.
- **Always show progress in plain words** ("Cleaning up your art…", "Laying stitches…").
- **Easy input.** Drag-and-drop, paste from clipboard, or file picker. Accept PNG, JPG,
  WEBP, SVG.
- **Errors say what to do next.** Never show a stack trace or error code to the user.
- Big buttons, readable text, works with a mouse alone.

## The user flow: 4 screens

1. **Upload.** Choose "Decorate a garment" or "Make a project" (in-the-hoop), then drop an
   image. The app asks two questions with picture buttons (for projects, the second
   question becomes the project type):
   - "How big will this be sewn?" Left chest 3.5", full front 10", hat front 4" × 2.25", or
     custom.
   - "What's it going on?" T-shirt, hoodie, polo, hat, towel, denim/canvas, other.

   These two answers drive both the AI cleanup and every stitch setting.
2. **Clean up (AI).** ChatGPT returns 3 embroidery-ready versions: Faithful, Simplified,
   Bold. Before/after slider on each.
   - User picks one, or types a change in plain words ("thicker outline", "drop the
     background", "only 4 colors").
   - Color chips show the final thread colors. Click to change or merge.
   - "Skip AI" uses the local cleanup only.
3. **Stitch.** The app digitizes automatically and shows the thread preview with a traffic
   light:
   - Green = ready. Yellow = warnings, each with a one-click fix. Red = will not sew
     well, with the reason.
   - Simple controls: size, number of colors, a "softer ↔ denser" slider.
   - Click any area to change its color, make it outline-only, or delete it.
4. **Download.** Pick the machine once (remembered after). The right file format is chosen
   automatically.
   - Downloads the stitch file plus a one-page PDF worksheet: preview, size, stitch
     count, color order with thread numbers, sew time, stabilizer to use.
   - Option to save to the project library.

## Under the hood: image → stitch file

Ten stages, each a separate module with its own tests and a debug image, so a bad result
can be traced to the step that caused it.

1. **Intake.** Auto-rotate, handle transparency, upscale small images, cap at 4096 px.
2. **Analyze.** OpenAI vision model returns JSON: subject type, key elements, any text
   (exact strings), color count, fine-detail risk. OpenCV measures dominant colors.
3. **Optimize.** ChatGPT image edit redraws the art for the chosen size and fabric (next
   section).
4. **Clean.** Local pass: reduce to N colors (k-means in Lab color space), remove specks
   below the smallest-detail size, smooth edges, close hairline gaps, merge near-identical
   colors.
5. **Vectorize.** Trace each color layer to paths (vtracer or potrace), simplify, scale to real
   millimeters.
6. **Classify.** Measure each shape's width: thin lines → running stitch, narrow columns →
   satin, wide areas → fill. Text is handled separately.
7. **Stitch.** Underlay first, then top stitches, using density, angle and compensation from
   the fabric preset.
8. **Sequence.** Order by color, background to foreground. Minimize color changes, jumps
   and trims. Hide travel stitches under later fills.
9. **Validate.** Check every rule in the Digitizing rules table and fit to hoop. Results feed
   the traffic light.
10. **Export.** Write the machine file, read it back to verify, render the preview, build the
    PDF worksheet.

## The ChatGPT stage

ChatGPT does the design work a human digitizer would do before stitching: simplify,
flatten colors, thicken lines, fix text. It uses the OpenAI API, which is pay-per-use and
separate from a ChatGPT subscription. It needs an API key with billing from
platform.openai.com, stored in local settings, never in code or GitHub.

### Calls

1. **Analyze** — Responses API, current vision-capable model, structured JSON output
   against a schema defined in code.
2. **Redraw** — Image API edits endpoint. As of Nov 2026, OpenAI's docs list
   `gpt-image-2.5-sunburst` (best for precise edits) and `gpt-image-2.5-flare` (faster).
   Default sunburst for final art, flare for quick previews. Model names go in config, never
   hard-coded — check current docs before building.
3. **Tweak** — the user's plain-words change is sent as a follow-up edit on the chosen
   version.
4. **Test first:** the image edit reference shows an SVG output option. If it gives cleaner
   vectors than tracing, use it and keep tracing as the fallback.

Redraw prompt template (variables filled from screen 1):

```
Redraw this image as embroidery-ready artwork for a {size_in}-inch design on {fabric}.
Rules: flat solid colors only, no gradients, shadows, textures or photo detail.
Maximum {n_colors} colors. Bold clean outlines at least {min_line_mm} mm thick at final size.
Remove any detail smaller than {min_detail_mm} mm at final size.
Keep text exactly as written: "{text}", at least {min_text_mm} mm tall.
Plain white background. Keep the subject, pose and layout recognizable. Center it.
```

The 3 versions change only the dials: Faithful (closest to original), Simplified (fewer colors
and details), Bold (thicker outlines, patch style).

### Guardrails

- **Text check.** Image models often misspell text. After every redraw, the vision model
  compares text to the original and flags changes: "AI changed your text — fix it?" For
  text-heavy designs, offer real embroidery lettering instead of AI-drawn text.
- **Cost control.** Show estimated cost per run, cache every result, low quality for
  previews and high for final, daily spend cap in settings.
- **Fallback.** No key, offline, or API error → local cleanup only. The app must always
  produce a file.

## Digitizing rules

These are standard starting values, to be tuned per fabric after real sew-outs. All live in
the Advanced drawer and the preset files.

| Rule | Default | Why |
|---|---|---|
| Smallest detail | 1 mm | Smaller won't sew or vanishes |
| Lines under 1.5 mm wide | Running stitch (triple/bean if bolder) | Too thin for satin |
| Satin column width | 1.5–7 mm | Wider satins snag and loop |
| Areas wider than 7 mm | Fill (tatami) | Stable over large areas |
| Smallest text | 5 mm capital height; simple sans fonts under 7 mm | Below this letters fill in |
| Fill and satin spacing | 0.4 mm | Coverage without stiffness |
| Fill stitch length | 3.5 mm | Smooth, even texture |
| Running stitch length | 2.5 mm; 1.5 mm on tight curves | Follows curves cleanly |
| Max stitch length | 7 mm; split anything longer | Long stitches snag |
| Min stitch length | 0.5 mm; drop shorter | Short stitches break thread |
| Pull compensation | +0.2 mm wovens, +0.4 mm knits | Stitches pull shapes narrower |
| Color overlap | 0.3–0.5 mm where colors meet | Prevents gaps |
| Underlay | Satin 2–4 mm: center run. Over 4 mm: edge run + zigzag. Fills: tatami underlay at 90° to top | Stabilizes fabric |
| Fill angle | Alternate between neighbors (e.g. 45° / 135°) | Adds depth, reduces puckering |
| Trim jumps longer than | 3 mm (setting) | Clean back, fewer loose threads |
| Default max colors | 6 | Fewer changes, cleaner look |

Stitch types: running, triple/bean, satin and tatami fill at launch. Later: contour fill,
blend/gradient fill, applique, 3D puff for hats.

## Presets: fabric, hoop, machine

The user picks a garment, not settings. Each fabric preset is a JSON file that sets density,
underlay and compensation, and puts the right stabilizer on the worksheet.

| Garment | Fill spacing | Underlay | Pull comp | Stabilizer on worksheet | Special handling |
|---|---|---|---|---|---|
| T-shirt (knit) | 0.45 mm | Heavy | +0.4 mm | Cutaway | Warn on large solid fills |
| Hoodie / sweatshirt | 0.42 mm | Medium-heavy | +0.35 mm | Cutaway | — |
| Polo (pique) | 0.42 mm | Medium | +0.35 mm | No-show mesh cutaway | — |
| Hat (structured) | 0.4 mm | Center run + edge run | +0.3 mm | Cap tearaway | Sew center-out, bottom-up; max height 2.25" (57 mm) |
| Towel / fleece | 0.4 mm | Heavy | +0.3 mm | Tearaway + water-soluble topping | Add knockdown stitch under design |
| Denim / canvas | 0.4 mm | Light | +0.2 mm | Tearaway | — |

**Hoops:** 4×4 (100×100 mm), 5×7 (130×180), 6×10 (160×260), 8×8 (200×200), 8×12
(200×300), cap frame, custom. Warn when the design comes within 5 mm of the hoop
edge.

**Machine → file format** (picked once, remembered):

| Machine brand | Format |
|---|---|
| Brother / Baby Lock | PES |
| Tajima and most commercial multi-needle | DST |
| Janome | JEF |
| Melco / Bernina | EXP |
| Husqvarna Viking / Pfaff | VP3 |
| Singer | XXX |

DST files store no thread colors, so the PDF worksheet always carries the color order.
Offer "download all formats" as a zip.

## What makes good embroidery, by material

The same file can't sew well on everything: the right density and underlay for felt are very
different from leather. So the garment picker drives the stitch settings automatically, and
these rules extend the presets above. Each row becomes a preset JSON file plus the
warnings the app shows.

| Material | Main risk | What the app changes | Stabilizer on worksheet |
|---|---|---|---|
| T-shirts, performance wear, polos | Fabric stretches, design warps or narrows | Much lower density, no-show mesh, avoid heavy solid designs; extra edge + zigzag underlay; higher pull comp | No-show mesh cutaway |
| Fleece / sweatshirt fleece | Pile swallows detail | Bolder minimum detail, more underlay, optional knockdown | Cutaway + spray adhesive; topper on dense designs |
| Towels, terry, minky | Loops poke through, text sinks | Auto-add a knockdown stitch: very low-density fill, no underlay, sewn before the design (e.g. 1.5 mm crosshatch), with a satin border to hide its edge; push users toward bigger designs | Tearaway or cutaway + water-soluble topping |
| Leather, faux leather, vinyl | Needle holes are permanent; dense stitching cuts it like a perforation | Open density 0.55–0.65 mm, no tiny details; longer stitches, minimal underlay | Cutaway or self-adhesive cutaway |
| Neoprene (koozies) | Spongy, compresses and distorts | Looser density, extra pull comp | Heavier backing |
| Silk, satin, delicate | Puckers, shows needle marks | Lower density, shorter stitches, minimal underlay | Light cutaway |
| Twill, canvas, denim, totes | Stable; heavy designs get stiff | Slightly reduced density to avoid thread buildup | Tearaway |

### Hats

Hats get their own sewing order, not just different numbers.

- **Center-out, bottom-up, always.** Sewing from the center outward pushes fabric
  evenly to the sides and prevents flagging and bird-nesting. The sequencer gets a
  dedicated cap mode for this.
- **Height limit.** Cap designs max out around 2.25" tall. The app warns and offers to
  resize.
- **Structured vs. unstructured.** Structured caps have a rigid buckram front that's ideal
  for detail; unstructured dad hats and bucket hats need extra stabilization. Worksheet
  says tearaway for structured, cutaway or doubled tearaway for unstructured.
- **Performance-fabric caps.** Some digitizers advise against them because of puckering
  and registration problems. The app shows a yellow warning.
- **Beanies.** Hoop flat, cutaway backing plus water-soluble topping.

### 3D puff (hats)

Puff is an option on structured caps only. Printful offers it only on hats because the foam
and density need a sturdy base.

- **Satin only, sized for foam.** Columns roughly 3–7 mm wide. Wilcom's Q&A says 3 mm
  minimum is best case; wider columns risk splitting and exposing foam. Wider areas
  get split into several columns.
- **Spacing.** At least 1.8 mm between elements so shapes don't merge as foam expands.
- **Closed ends.** Every column end is capped so foam can't peek out; a walk stitch tacks
  the foam first.
- **Push comp** around 0.17–0.25 mm for puff.
- **Foam:** files assume standard 2–3 mm foam; the worksheet adds a "place foam now"
  stop.
- The AI redraw gets a puff variant: bold block letters, rounded corners, no thin strokes.

New garment presets to add: fleece, towel/minky, leather/vinyl, neoprene, delicate,
beanie, structured cap, unstructured cap, 3D puff cap.

## In-the-hoop projects

In-the-hoop (ITH) means the machine builds a standalone item inside the hoop instead of
just decorating fabric. The app gets a second mode, "Make a project", next to "Decorate a
garment." The user picks a project type, uploads their art, and the app builds the
construction steps around it, plus cutting templates and printed instructions.

### How every ITH file works

Every project follows the same pattern: a placement line, a tackdown line, then detailing
and a finishing outline. The key technical rule: machines only reliably stop at a color
change, so **every construction step must be its own color in the file**, even if it's the same
thread. The placement line sews first on bare stabilizer, then fabric goes on top of it.

Step types the generator must build:

| Step | What it sews | Machine stops so the user can… |
|---|---|---|
| Placement line (dieline) | Single run on the shape's outline | Lay fabric over it |
| Tackdown | Run just inside the placement line | Trim excess fabric |
| Decoration | The user's digitized art, clipped to stay inside the shape | Change thread colors |
| Back-fabric tackdown | Run after fabric is placed on the back of the hoop | Flip the hoop and place backing |
| Zipper tackdown | Runs on both sides of the zipper teeth | Tape the zipper in place first |
| Final seam | Triple run around the shape, with a turning gap (stitch-and-turn) | Unhoop and finish |
| Finishing border | Satin border over raw edges (vinyl, felt, lace edges) | Unhoop and finish |

Extra outputs per project:

- **Dieline templates.** ITH designs ship with separate dieline files for cutting fabric to
  shape. The app exports them as stitch files and as a printable PDF at 100% scale.
- **Step-by-step instruction PDF**, built automatically from the step list: a preview picture
  per step, what to place, what to trim. Thread chart labels steps by name ("Step 1 —
  Placement line, any color").
- **Multi-hoop split.** When parts don't fit, the app splits them into "a" and "b" files, the
  way small plush appendages sew in a separate file.

### Project templates to ship

| Project | How it's built | Hoop | Notes |
|---|---|---|---|
| Snap-tab key fob / bag tag | Vinyl front → decoration → vinyl back on hoop back → border → trim, punch and set snap | 4×4 or 5×7 (eyelet fobs 4×4, snap tabs 5×7) | Tab about 0.75" wide; use non-fray vinyl, faux leather or felt |
| Felties / patches | Placement → felt tackdown → decoration → satin border | 4×4 | Easiest first project; good for testing |
| Coasters and ornaments (stitch-and-turn) | Dieline → front fabric → decoration → back fabric → final seam with gap | 4×4–5×7 | Trim to a ¼" seam, turn, stuff if needed, hand-close the gap; clip corners before turning |
| Plush toys (stitch-and-turn) | Same as ornaments; ears/arms sewn first as small pieces and placed before the final seam | 5×7 and up; big plush = multi-hoop | Lightweight fleece, flannel or microsuede for the body; minky needs water-soluble topper; faces are embroidered |
| Lined zipper bags | Zipper placement + tackdown → outer and lining on both sides → final seam → turn through zipper | 6×10 or larger for lined bags | No-show mesh cutaway, not tearaway; keep batting out of the seam allowance |
| Pillows | Small pillows fully ITH with an overlapped envelope back. Bigger pillows: app exports the embroidered front panel + a cutting/sewing sheet for the envelope back | Panel limited by hoop (8×12 max ≈ 7½×11½" face) | Big pillows = panel mode, finished on a sewing machine |
| Freestanding lace (ornaments, bookmarks, earrings) | Base grid of two fill layers at 90°, then decoration and a satin frame, on wash-away stabilizer | 4×4–5×7 | Much higher density than normal; any visible gap means it falls apart; grid made by crossing stitch angles; same thread in the bobbin |

### ITH generator rules (defaults, editable)

- **Shape:** from the art's outline, offset outward, or from the template library.
- **Seam allowance:** ¼" (6 mm) outside the final seam, built into the dieline templates.
- **Keep-out zone:** decoration stays at least 6 mm inside the final seam so it doesn't get
  caught when turning.
- **Turning gap:** 2–3", scaled to the item, placed on a straight edge.
- **Fit check:** the whole project, tabs included, must fit inside the hoop with a 5 mm
  margin.
- **FSL check:** the validator flags any open spots in the lace base before export.
- **Kids' toys:** default plush faces to embroidered eyes, no hardware. Toys graded under
  age 3 need small-parts testing at a CPSC-accepted lab, even for registered small
  batch makers.

## Preview and export

The preview has to look like real thread, because that's how the user judges the file before
wasting a garment.

- **Thread preview.** Draw every stitch as a thread-thick line with light sheen, on the
  garment color the user picks. Zoom to 1:1 real size.
- **Sew-out simulator.** Play button animates the sewing order with a scrub bar. Color
  changes and trims are marked.
- **Stats.** Size (in and mm), stitch count, colors, trims, sew time (stitches ÷ machine
  speed, default 650 per minute, editable).
- **Thread matching.** Map each color to the nearest thread in the chosen brand (Madeira
  Polyneon, Isacord, Robison-Anton, Brother) by color distance in Lab space. User can
  override. Thread charts ship as JSON.
- **Exports:** stitch file(s), PDF worksheet, PNG mockup on the garment (for customer
  approval and listings), and a project file to reopen and edit later.
- **Self-check.** Read every exported file back with pyembroidery and confirm stitch
  count and size match before offering the download.

## Tech stack and architecture

Python backend because the embroidery and image libraries are Python; browser front
end because it's the easiest UI to make friendly.

- **Backend:** Python 3.12 + FastAPI.
- **Libraries:** pyembroidery (machine file read/write, MIT license; Ink/Stitch is renaming
  its copy to pystitch, so pin the version), OpenCV + scikit-image (cleanup, color
  reduction), vtracer or potrace (tracing), shapely (geometry: widths, offsets, fill lines),
  numpy, official openai SDK, WeasyPrint or reportlab (PDF).
- **Front end:** single-page app (React + Vite, or Claude Code's simpler pick), served by
  the backend as one bundle. Canvas or WebGL for the thread preview.
- **Runs locally:** double-click launcher (start.bat, later a PyInstaller .exe) opens the
  browser at localhost.
- **Updates from GitHub** (BookWorm824) on launch, like her other apps.
- **Settings file:** API key, model names, spend cap, default machine, default thread brand.

Folder layout:

```
app/api          — endpoints
app/pipeline/    — one module per stage: intake, analyze, optimize, clean,
                   vectorize, classify, stitch, sequence, validate, export
app/presets/ and app/threads/ — JSON
web/             — front end
tests/images/    — test images + expected outputs
```

**Later, hosted:** same backend in Docker. Add accounts and billing only then.

## Build phases

Stitch engine first, AI second: if the stitches are bad, pretty art won't save them. Every
phase ends with a real sew-out before the next starts.

1. **Phase 1 — Stitch engine (no AI).** Clean PNG logo → color reduction → trace → fill +
   running stitch → PES and DST export → basic preview. Goal: first sew-out on a test tee.
2. **Phase 2 — AI cleanup.** Analysis, redraw, 3 versions, tweak box, text check, local
   fallback.
3. **Phase 3 — Stitch quality.** Satin columns, underlay, pull compensation, fabric presets,
   smart sequencing and trims, traffic-light validator.
4. **Phase 4 — Friendly polish.** Final 4-screen wizard, click-to-edit areas, thread preview +
   simulator, thread brand matching, PDF worksheet, PNG mockup, project library.
5. **Phase 5 — In-the-hoop projects.** "Make a project" mode, ITH step generator, dieline
   templates, instruction PDF. Build in this order: felties → key fobs →
   coasters/ornaments → plush → zipper bags → pillows → freestanding lace. Each type
   gets a sew-out before the next.
6. **Phase 6 — Extras.** Lettering tool with embroidery fonts, 3D puff mode, batch mode,
   .exe launcher with GitHub auto-update.

## Definition of done

The app is done when all of these pass:

- Test set of 12 images in `tests/images`: 3 simple logos, 3 cartoon/clipart, 2 text-heavy,
  2 photos (pet, person), 2 with transparent backgrounds.
- Every test image goes from upload to file in 4 screens or fewer using only defaults.
- Every exported file opens without errors in a second program (Ink/Stitch or the
  machine maker's free viewer) and reads back with matching stitch count.
- Validator passes on every file: no stitch over max length, no detail under minimum, fits
  the chosen hoop.
- Text in the output matches the original spelling.
- The app still produces a file with no API key.
- Someone who has never digitized makes a file without help.
- Lauren sews 3 designs on a tee, a hat and a towel: no gaps between colors, no
  puckering, no thread breaks or bird nests, text readable.
- Each material preset (fleece, towel, leather/vinyl, structured cap, 3D puff) gets one
  test sew-out that passes the same checks.
- Each ITH template produces a finished item by following only the app's instruction
  PDF: machine stops at every step, dieline templates cut to the right size, item turns
  and closes cleanly.

## Honest limits, costs and licensing

- **No auto-digitizer is perfect.** Even pro software needs touch-up after auto-digitizing.
  Simple logos and cartoon art should come out sew-ready; photos and tiny detail will
  always be simplified. The AI redraw is what closes the gap: clean flat art in, good
  stitches out.
- **Sew-outs are the real test.** Budget test garments and thread for tuning the presets.
- **API cost.** Each design is 1 analysis call + 3 image edits, plus tweaks. Billed per use,
  separate from ChatGPT. Check OpenAI's current pricing before launch; the app shows
  cost per run and enforces the spend cap.
- **Licensing.** pyembroidery is MIT: free to use in anything. Ink/Stitch is GPL-3: a great
  reference for satin and fill-routing algorithms, but copying its code makes the whole
  app GPL. Fine for personal use; it matters if the app is ever sold as downloadable
  software. Rule for Claude Code: write our own stitch code, read Ink/Stitch for ideas
  only, and flag any borrowed code.

## Open questions for Lauren

Answer these before Phase 1; the brief assumes the defaults in brackets.

- Which embroidery machine(s)? Brand, model, single- or multi-needle, hoop sizes. [PES
  + DST first, 6 colors max]
- Shop tool only, or a product to sell later? [Local tool first, hosted later]
- Monthly API spend cap? [Set in settings, default low]
- Preferred thread brand? [Madeira Polyneon]
- Largest hoop you have? It decides which ITH projects fit in one hooping (lined zipper
  bags need 6×10 or larger). [Plan for 5×7 and 6×10]
- Which ITH projects first? [Felties and key fobs, then plush and zipper bags]

## Kickoff prompt for Claude Code

Export this doc as Markdown, save it as `BRIEF.md` in an empty project folder, open
Claude Code there, and paste:

```
Read BRIEF.md in this folder. It is the full spec for an AI embroidery digitizer.
Build it in the phases listed under "Build phases". Start with Phase 1 only.
Before writing code:
1. Check the current OpenAI API docs for image model names and the image edit endpoint,
   and the current pyembroidery / pystitch release.
2. Give me a short plan: folder structure and stack. Wait for my OK.
Rules:
- Extremely simple to use: plain words, smart defaults, no jargon on main screens.
- Never put my API key in code or commit it to GitHub.
- Set up the repo under my GitHub (BookWorm824) so the app updates itself from GitHub.
- Write our own stitch code. Use Ink/Stitch for ideas only and flag anything borrowed.
After Phase 1, give me a test file to sew out and tell me what to check,
one step at a time, in plain language.
```
