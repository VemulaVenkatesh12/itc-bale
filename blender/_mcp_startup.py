"""One-time startup script: enables the BlenderMCP addon and starts its
socket server so a live Blender GUI session is ready for MCP connections.
Run via: blender --python _mcp_startup.py  (foreground GUI, not --background)
"""
import bpy

ADDON_MODULE = "blender_mcp"

bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
print(f"[mcp_startup] addon '{ADDON_MODULE}' enabled")

scene = bpy.context.scene
scene.blendermcp_port = 9876
bpy.ops.blendermcp.start_server()
print(f"[mcp_startup] server running: {scene.blendermcp_server_running} on port {scene.blendermcp_port}")
