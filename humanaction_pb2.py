# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: humanaction.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='humanaction.proto',
  package='humanaction',
  syntax='proto3',
  serialized_options=_b('\n\033io.grpc.examples.helloworldB\017HelloWorldProtoP\001\242\002\003HLW'),
  serialized_pb=_b('\n\x11humanaction.proto\x12\x0bhumanaction\"\x18\n\x05\x43hunk\x12\x0f\n\x07\x43ontent\x18\x01 \x01(\x0c\"\xdd\x02\n\x05label\x12\x10\n\x08message1\x18\x01 \x01(\x02\x12\x10\n\x08message2\x18\x02 \x01(\x02\x12\x10\n\x08message3\x18\x03 \x01(\x02\x12\x10\n\x08message4\x18\x04 \x01(\x02\x12\x10\n\x08message5\x18\x05 \x01(\x02\x12\x10\n\x08message6\x18\x06 \x01(\x02\x12\x10\n\x08message7\x18\x07 \x01(\x02\x12\x10\n\x08message8\x18\x08 \x01(\x02\x12\x10\n\x08message9\x18\t \x01(\x02\x12\x11\n\tmessage10\x18\n \x01(\x02\x12\x0e\n\x06\x63lass1\x18\x0b \x01(\t\x12\x0e\n\x06\x63lass2\x18\x0c \x01(\t\x12\x0e\n\x06\x63lass3\x18\r \x01(\t\x12\x0e\n\x06\x63lass4\x18\x0e \x01(\t\x12\x0e\n\x06\x63lass5\x18\x0f \x01(\t\x12\x0e\n\x06\x63lass6\x18\x10 \x01(\t\x12\x0e\n\x06\x63lass7\x18\x11 \x01(\t\x12\x0e\n\x06\x63lass8\x18\x12 \x01(\t\x12\x0e\n\x06\x63lass9\x18\x13 \x01(\t\x12\x0f\n\x07\x63lass10\x18\x14 \x01(\t2G\n\x0bHumanAction\x12\x38\n\x08\x43lassify\x12\x12.humanaction.Chunk\x1a\x12.humanaction.label\"\x00(\x01\x30\x01\x42\x36\n\x1bio.grpc.examples.helloworldB\x0fHelloWorldProtoP\x01\xa2\x02\x03HLWb\x06proto3')
)




_CHUNK = _descriptor.Descriptor(
  name='Chunk',
  full_name='humanaction.Chunk',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Content', full_name='humanaction.Chunk.Content', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=58,
)


_LABEL = _descriptor.Descriptor(
  name='label',
  full_name='humanaction.label',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message1', full_name='humanaction.label.message1', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message2', full_name='humanaction.label.message2', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message3', full_name='humanaction.label.message3', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message4', full_name='humanaction.label.message4', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message5', full_name='humanaction.label.message5', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message6', full_name='humanaction.label.message6', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message7', full_name='humanaction.label.message7', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message8', full_name='humanaction.label.message8', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message9', full_name='humanaction.label.message9', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message10', full_name='humanaction.label.message10', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class1', full_name='humanaction.label.class1', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class2', full_name='humanaction.label.class2', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class3', full_name='humanaction.label.class3', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class4', full_name='humanaction.label.class4', index=13,
      number=14, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class5', full_name='humanaction.label.class5', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class6', full_name='humanaction.label.class6', index=15,
      number=16, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class7', full_name='humanaction.label.class7', index=16,
      number=17, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class8', full_name='humanaction.label.class8', index=17,
      number=18, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class9', full_name='humanaction.label.class9', index=18,
      number=19, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class10', full_name='humanaction.label.class10', index=19,
      number=20, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=61,
  serialized_end=410,
)

DESCRIPTOR.message_types_by_name['Chunk'] = _CHUNK
DESCRIPTOR.message_types_by_name['label'] = _LABEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Chunk = _reflection.GeneratedProtocolMessageType('Chunk', (_message.Message,), dict(
  DESCRIPTOR = _CHUNK,
  __module__ = 'humanaction_pb2'
  # @@protoc_insertion_point(class_scope:humanaction.Chunk)
  ))
_sym_db.RegisterMessage(Chunk)

label = _reflection.GeneratedProtocolMessageType('label', (_message.Message,), dict(
  DESCRIPTOR = _LABEL,
  __module__ = 'humanaction_pb2'
  # @@protoc_insertion_point(class_scope:humanaction.label)
  ))
_sym_db.RegisterMessage(label)


DESCRIPTOR._options = None

_HUMANACTION = _descriptor.ServiceDescriptor(
  name='HumanAction',
  full_name='humanaction.HumanAction',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=412,
  serialized_end=483,
  methods=[
  _descriptor.MethodDescriptor(
    name='Classify',
    full_name='humanaction.HumanAction.Classify',
    index=0,
    containing_service=None,
    input_type=_CHUNK,
    output_type=_LABEL,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_HUMANACTION)

DESCRIPTOR.services_by_name['HumanAction'] = _HUMANACTION

# @@protoc_insertion_point(module_scope)
