"""A stand-in for NVIDIA kaolin in the TriFlow process (MIT, Spellcast3D's own code; not kaolin's).

TriFlow imports TRELLIS, whose mesh representation imports FlexiCubes, which does
`from kaolin.utils.testing import check_tensor` when it loads. TriFlow's inference never runs
FlexiCubes, so the real kaolin (plus warp-lang, usd-core and a Jupyter stack, ~200 MB, partly
under NVIDIA's non-commercial license) is not installed: launchers/triflow_server.py puts this
folder on the path instead. Only kaolin.utils.testing.check_tensor exists here.
"""
