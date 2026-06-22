# Orbital 3D demo

The standalone, interactive Three.js orbit visualization lives at
`threejs_orbit/orbit_scene.html`, not in this `examples/` folder.

That file is fully self-contained: Three.js r128 and the trained orbit
data are both embedded inline, so it opens directly in any browser with
no server, no CDN, and no network access required.

An earlier `orbit_demo.html` in this folder referenced
`../node_modules/three/build/three.core.min.js` as an external script —
that build is an ES module (uses `export` statements) and cannot be
loaded as a plain `<script src="...">` tag, so the file threw
`Uncaught SyntaxError: Unexpected token 'export'` immediately on load
and never rendered anything. It has been removed rather than left as a
broken artifact. See `threejs_orbit/orbit_scene.html` for the working
version, which has been verified end-to-end with Playwright (zero
console errors, confirmed canvas rendering, tested drag/scrub/animate
interactions).

To regenerate the underlying orbit data yourself:

```bash
python examples/generate_orbit_data.py
```
