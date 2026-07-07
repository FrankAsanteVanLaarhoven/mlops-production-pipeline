import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf.descriptor import FieldDescriptor
    FieldDescriptor.label = property(lambda self: 3 if self.is_repeated else (2 if self.is_required else 1))
    print("[sitecustomize] Protobuf FieldDescriptor monkeypatch applied successfully!")
except Exception as e:
    print(f"[sitecustomize] Failed to apply monkeypatch: {e}")
