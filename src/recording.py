"""Persist live Crafter frames and portable replay evidence."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.agent import EpisodeFrame, EpisodeTrace

VIEWER_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Crafter gameplay</title>
<style>
body{background:#111827;color:#f3f4f6;font:16px system-ui;margin:0;padding:24px}
main{max-width:1050px;margin:auto}section{display:flex;gap:24px;flex-wrap:wrap}
img{width:min(512px,100%);image-rendering:pixelated;border:1px solid #64748b}
article{flex:1;min-width:260px}pre{white-space:pre-wrap;font-size:13px}
input{width:min(512px,100%)}button{padding:8px 12px;margin:12px 8px 12px 0}
#label{color:#fbbf24}a{color:#7dd3fc}
</style><main><h1>Crafter gameplay</h1><p id="label">Loading…</p><section><div>
<img id="frame" alt="Rendered Crafter frame"><br><button id="follow">Follow live</button>
<button id="play">Play replay</button><br><input id="seek" type="range" min="0" max="0" value="0">
<p id="status"></p><a href="replay.gif">GIF</a> · <a href="result.json">Trace</a></div>
<article><h2 id="step">Step 0</h2><p id="action"></p><pre id="observation"></pre></article>
</section></main><script>
const q=id=>document.getElementById(id);let data=null,pos=0,follow=true,playing=false;
function show(index){if(!data?.frames.length)return;pos=Math.max(0,Math.min(index,data.frames.length-1));
 const f=data.frames[pos];q('frame').src=f.path+'?v='+Date.now();q('step').textContent='Step '+f.step;
 q('action').textContent=f.action?'Action: '+f.action:'Initial world';q('observation').textContent=f.observation;
 q('seek').value=pos;q('status').textContent=`${data.state} · ${pos+1}/${data.frames.length}`+
 (data.termination_reason?' · '+data.termination_reason:'');}
async function refresh(){try{const r=await fetch('live.json',{cache:'no-store'});data=await r.json();
 q('label').textContent=data.label+' · '+(data.is_stub?'OFFLINE STUB':'LIVE LLM');q('seek').max=Math.max(0,data.frames.length-1);
 show(follow?data.frames.length-1:pos);}catch{q('status').textContent='Serve this directory over HTTP to follow live.';}}
q('follow').onclick=()=>{follow=true;playing=false;refresh()};q('seek').oninput=()=>{follow=false;playing=false;show(+q('seek').value)};
q('play').onclick=()=>{follow=false;playing=!playing;if(playing&&pos>=data.frames.length-1)show(0)};
setInterval(()=>{if(playing&&data){if(pos<data.frames.length-1)show(pos+1);else playing=false}},200);
refresh();setInterval(refresh,250);
</script></html>"""


class EpisodeRecorder:
    """Write reset/post-action frames without replacing an existing run."""

    def __init__(self, path: Path, *, label: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir()
        (self.path / "index.html").write_text(VIEWER_HTML)
        self.label = label
        self.frames: list[dict] = []
        self.is_stub = True
        self._publish("waiting")

    def _write_json(self, name: str, payload: dict) -> None:
        target = self.path / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(target)

    def _publish(self, state: str, termination_reason: str | None = None) -> None:
        self._write_json(
            "live.json",
            {
                "schema_version": 1,
                "label": self.label,
                "is_stub": self.is_stub,
                "state": state,
                "latest": self.frames[-1] if self.frames else None,
                "frames": self.frames,
                "termination_reason": termination_reason,
            },
        )

    def on_frame(self, frame: EpisodeFrame) -> None:
        """Encode a frame before exposing it to polling viewers."""
        relative_path = f"frames/{frame.step:05d}.png"
        Image.fromarray(frame.rgb).save(self.path / relative_path)
        self.is_stub = frame.is_stub
        self.frames.append(
            {
                "step": frame.step,
                "action": frame.action,
                "path": relative_path,
                "observation": frame.observation,
                "achievements": frame.achievements,
                "inventory": frame.inventory,
            }
        )
        self._publish("playing")

    def finish(self, trace: EpisodeTrace) -> None:
        """Write the trace and GIF before marking the run complete."""
        self._write_json("result.json", trace.to_dict())
        images: list[Image.Image] = []
        try:
            for frame in self.frames:
                with Image.open(self.path / frame["path"]) as image:
                    images.append(image.copy())
            if images:
                images[0].save(
                    self.path / "replay.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=200,
                    loop=0,
                    optimize=False,
                )
        finally:
            for image in images:
                image.close()
        self._publish("finished", trace.termination_reason)

    def fail(self, reason: str) -> None:
        """Leave partial evidence visible after interruption."""
        self._publish("interrupted", reason)
