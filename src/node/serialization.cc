/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file node/serialization.cc
 * \brief Utilities to serialize TVM AST/IR objects.
 */
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/container.h>
#include <tvm/node/reflection.h>
#include <tvm/node/serialization.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

// <bojian/TVM-AutoDiff> Newly-Added Headers
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <dmlc/parameter.h>  // GetEnv
#include <sstream>
#include "../runtime/object_internal.h"

#include <cctype>
#include <map>
#include <string>

#include "../support/base64.h"

namespace tvm {

inline std::string Type2String(const DataType& t) { return runtime::DLDataType2String(t); }

inline DataType String2Type(std::string s) { return DataType(runtime::String2DLDataType(s)); }

inline std::string Base64Decode(std::string s) {
  dmlc::MemoryStringStream mstrm(&s);
  support::Base64InStream b64strm(&mstrm);
  std::string output;
  b64strm.InitPosition();
  dmlc::Stream* strm = &b64strm;
  strm->Read(&output);
  return output;
}

inline std::string Base64Encode(std::string s) {
  std::string blob;
  dmlc::MemoryStringStream mstrm(&blob);
  support::Base64OutStream b64strm(&mstrm);
  dmlc::Stream* strm = &b64strm;
  strm->Write(s);
  b64strm.Finish();
  return blob;
}

// indexer to index all the nodes
class NodeIndexer : public AttrVisitor {
 public:
  std::unordered_map<Object*, size_t> node_index_{{nullptr, 0}};
  std::vector<Object*> node_list_{nullptr};
  std::unordered_map<DLTensor*, size_t> tensor_index_;
  std::vector<DLTensor*> tensor_list_;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, void** value) final {}
  void Visit(const char* key, DataType* value) final {}

  void Visit(const char* key, runtime::NDArray* value) final {
    DLTensor* ptr = const_cast<DLTensor*>((*value).operator->());
    if (tensor_index_.count(ptr)) return;
    CHECK_EQ(tensor_index_.size(), tensor_list_.size());
    tensor_index_[ptr] = tensor_list_.size();
    tensor_list_.push_back(ptr);
  }

  void Visit(const char* key, ObjectRef* value) final {
    MakeIndex(const_cast<Object*>(value->get()));
  }

  // make index of all the children of node
  void MakeIndex(Object* node) {
    if (node == nullptr) return;
    CHECK(node->IsInstance<Object>());

    if (node_index_.count(node)) return;
    CHECK_EQ(node_index_.size(), node_list_.size());
    node_index_[node] = node_list_.size();
    node_list_.push_back(node);

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (const auto& sp : *n) {
        MakeIndex(const_cast<Object*>(sp.get()));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      bool is_str_map = std::all_of(n->begin(), n->end(), [](const auto& v) {
        return v.first->template IsInstance<StringObj>();
      });
      if (is_str_map) {
        for (const auto& kv : *n) {
          MakeIndex(const_cast<Object*>(kv.second.get()));
        }
      } else {
        for (const auto& kv : *n) {
          MakeIndex(const_cast<Object*>(kv.first.get()));
          MakeIndex(const_cast<Object*>(kv.second.get()));
        }
      }
    } else {
      // if the node already have repr bytes, no need to visit Attrs.
      if (!reflection_->GetReprBytes(node, nullptr)) {
        reflection_->VisitAttrs(node, this);
      }
    }
  }
};

// use map so attributes are ordered.
using AttrMap = std::map<std::string, std::string>;

/*! \brief Node structure for json format. */
struct JSONNode {
  /*! \brief The type of key of the object. */
  std::string type_key;
  /*! \brief The str repr representation. */
  std::string repr_bytes;
  /*! \brief the attributes */
  AttrMap attrs;
  /*! \brief keys of a map. */
  std::vector<std::string> keys;
  /*! \brief values of a map or array. */
  std::vector<size_t> data;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("type_key", type_key);
    if (repr_bytes.size() != 0) {
      // choose to use str representation or base64, based on whether
      // the byte representation is printable.
      if (std::all_of(repr_bytes.begin(), repr_bytes.end(),
                      [](char ch) { return std::isprint(ch); })) {
        writer->WriteObjectKeyValue("repr_str", repr_bytes);
      } else {
        writer->WriteObjectKeyValue("repr_b64", Base64Encode(repr_bytes));
      }
    }
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    if (keys.size() != 0) {
      writer->WriteObjectKeyValue("keys", keys);
    }
    if (data.size() != 0) {
      writer->WriteObjectKeyValue("data", data);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    attrs.clear();
    data.clear();
    repr_bytes.clear();
    type_key.clear();
    std::string repr_b64, repr_str;
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareOptionalField("type_key", &type_key);
    helper.DeclareOptionalField("repr_b64", &repr_b64);
    helper.DeclareOptionalField("repr_str", &repr_str);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.DeclareOptionalField("keys", &keys);
    helper.DeclareOptionalField("data", &data);
    helper.ReadAllFields(reader);

    if (repr_str.size() != 0) {
      CHECK_EQ(repr_b64.size(), 0U);
      repr_bytes = std::move(repr_str);
    } else if (repr_b64.size() != 0) {
      repr_bytes = Base64Decode(repr_b64);
    }
  }
};

// Helper class to populate the json node
// using the existing index.
class JSONAttrGetter : public AttrVisitor {
 public:
  const std::unordered_map<Object*, size_t>* node_index_;
  const std::unordered_map<DLTensor*, size_t>* tensor_index_;
  JSONNode* node_;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  void Visit(const char* key, double* value) final {
    std::ostringstream s;
    // Type <double> have approximately 16 decimal digits
    s.precision(16);
    s << (*value);
    node_->attrs[key] = s.str();
  }
  void Visit(const char* key, int64_t* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, uint64_t* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, int* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, bool* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, std::string* value) final { node_->attrs[key] = *value; }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "not allowed to serialize a pointer";
  }
  void Visit(const char* key, DataType* value) final { node_->attrs[key] = Type2String(*value); }

  void Visit(const char* key, runtime::NDArray* value) final {
    node_->attrs[key] =
        std::to_string(tensor_index_->at(const_cast<DLTensor*>((*value).operator->())));
  }

  void Visit(const char* key, ObjectRef* value) final {
    node_->attrs[key] = std::to_string(node_index_->at(const_cast<Object*>(value->get())));
  }

  // Get the node
  void Get(Object* node) {
    if (node == nullptr) {
      node_->type_key.clear();
      return;
    }
    node_->type_key = node->GetTypeKey();
    // do not need to print additional things once we have repr bytes.
    if (reflection_->GetReprBytes(node, &(node_->repr_bytes))) return;

    // populates the fields.
    node_->attrs.clear();
    node_->data.clear();

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (size_t i = 0; i < n->size(); ++i) {
        node_->data.push_back(node_index_->at(const_cast<Object*>(n->at(i).get())));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      bool is_str_map = std::all_of(n->begin(), n->end(), [](const auto& v) {
        return v.first->template IsInstance<StringObj>();
      });
      if (is_str_map) {
        for (const auto& kv : *n) {
          node_->keys.push_back(Downcast<String>(kv.first));
          node_->data.push_back(node_index_->at(const_cast<Object*>(kv.second.get())));
        }
      } else {
        for (const auto& kv : *n) {
          node_->data.push_back(node_index_->at(const_cast<Object*>(kv.first.get())));
          node_->data.push_back(node_index_->at(const_cast<Object*>(kv.second.get())));
        }
      }
    } else {
      // recursively index normal object.
      reflection_->VisitAttrs(node, this);
    }
  }
};

// Helper class to set the attributes of a node
// from given json node.
class JSONAttrSetter : public AttrVisitor {
 public:
  const std::vector<ObjectPtr<Object>>* node_list_;
  const std::vector<runtime::NDArray>* tensor_list_;
  JSONNode* node_;

  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  std::string GetValue(const char* key) const {
    auto it = node_->attrs.find(key);
    if (it == node_->attrs.end()) {
      LOG(FATAL) << "JSONReader: cannot find field " << key;
    }
    return it->second;
  }
  template <typename T>
  void ParseValue(const char* key, T* value) const {
    std::istringstream is(GetValue(key));
    is >> *value;

    // <bojian/TVM-AutoDiff> Removed the check on field format. The reason is
    //                       because the type for tir.IterVar has been changed
    //                       from std::string to String.
    // if (is.fail()) {
    //   LOG(FATAL) << "Wrong value format for field " << key;
    // }
  }
  void Visit(const char* key, double* value) final { ParseValue(key, value); }
  void Visit(const char* key, int64_t* value) final { ParseValue(key, value); }
  void Visit(const char* key, uint64_t* value) final { ParseValue(key, value); }
  void Visit(const char* key, int* value) final { ParseValue(key, value); }
  void Visit(const char* key, bool* value) final { ParseValue(key, value); }
  void Visit(const char* key, std::string* value) final { *value = GetValue(key); }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "not allowed to deserialize a pointer";
  }
  void Visit(const char* key, DataType* value) final {
    std::string stype = GetValue(key);
    *value = String2Type(stype);
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, tensor_list_->size());
    *value = tensor_list_->at(index);
  }
  void Visit(const char* key, ObjectRef* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, node_list_->size());
    *value = ObjectRef(node_list_->at(index));
  }
  // set node to be current JSONNode
  // <bojian/TVM-AutoDiff> Fix for MapNode construction.
  // void Set(Object* node) {
  void Set(ObjectPtr<Object>* pnode) {
    Object* const node = pnode->get();
    if (node == nullptr) return;

    // <bojian/TVM-AutoDiff> Added logging on the JSON node type key.
    // LOG(INFO) << "Type Key: " << node_->type_key;
    // <bojian/TVM-AutoDiff> Added the special handling for StrMap data structure.
    // if (node_->type_key == "StrMap") {
    //   LOG(INFO) << "StrMap data structure encountered"; 
    //   MapNode* n = static_cast<MapNode*>(node);
    //   for (const auto& kv : n->data) {
    //     node_->keys.push_back(Downcast<String>(kv.first));
    //     node_->data.push_back(node_index_->at(const_cast<Object*>(kv.second.get())));
    //   }
    // }

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      CHECK_EQ(n->size(), node_->data.size());
      int64_t i = 0;
      for (size_t index : node_->data) {
        n->SetItem(i++, ObjectRef(node_list_->at(index)));
      }
    } else if (node->IsInstance<MapNode>()) {

      // <bojian/TVM-AutoDiff> Fix for MapNode construction.
      // MapNode* n = static_cast<MapNode*>(node);
      std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual> container;

      if (node_->keys.empty()) {
        CHECK_EQ(node_->data.size() % 2, 0U);
        for (size_t i = 0; i < node_->data.size(); i += 2) {

          // <bojian/TVM-AutoDiff> Fix for MapNode construction.
          // (*n).at(ObjectRef(node_list_->at(node_->data[i]))) =
          //         ObjectRef(node_list_->at(node_->data[i + 1]));
          container[ObjectRef(node_list_->at(node_->data[i]))] =
                    ObjectRef(node_list_->at(node_->data[i + 1]));
        }
      } else {
        CHECK_EQ(node_->data.size(), node_->keys.size());
        for (size_t i = 0; i < node_->data.size(); ++i) {

          // <bojian/TVM-AutoDiff> Fix for MapNode construction.
          // (*n).at(String(node_->keys[i])) = ObjectRef(node_list_->at(node_->data[i]));
          container[String(node_->keys[i])]
              = ObjectRef(node_list_->at(node_->data[i]));
        }
      }  // if (node_->keys.empty())

      // <bojian/TVM-AutoDiff> Fix for MapNode construction.
      Map<ObjectRef, ObjectRef> map(container);
      *pnode = ::tvm::runtime::ObjectInternal::MoveObjectPtr(&map);

    } else {
      reflection_->VisitAttrs(node, this);
    }  // if (node->IsInstance<ArrayNode>())
  }
};

// json graph structure to store node
struct JSONGraph {
  // the root of the graph
  size_t root;
  // the nodes of the graph
  std::vector<JSONNode> nodes;
  // base64 b64ndarrays of arrays
  std::vector<std::string> b64ndarrays;
  // global attributes
  AttrMap attrs;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("root", root);
    writer->WriteObjectKeyValue("nodes", nodes);
    writer->WriteObjectKeyValue("b64ndarrays", b64ndarrays);
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    attrs.clear();
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("root", &root);
    helper.DeclareField("nodes", &nodes);
    helper.DeclareOptionalField("b64ndarrays", &b64ndarrays);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }

  static JSONGraph Create(const ObjectRef& root) {
    JSONGraph g;
    NodeIndexer indexer;
    indexer.MakeIndex(const_cast<Object*>(root.get()));
    JSONAttrGetter getter;
    getter.node_index_ = &indexer.node_index_;
    getter.tensor_index_ = &indexer.tensor_index_;
    for (Object* n : indexer.node_list_) {
      JSONNode jnode;
      getter.node_ = &jnode;
      getter.Get(n);
      g.nodes.emplace_back(std::move(jnode));
    }
    g.attrs["tvm_version"] = TVM_VERSION;
    g.root = indexer.node_index_.at(const_cast<Object*>(root.get()));
    // serialize tensor
    for (DLTensor* tensor : indexer.tensor_list_) {
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      support::Base64OutStream b64strm(&mstrm);
      runtime::SaveDLTensor(&b64strm, tensor);
      b64strm.Finish();
      g.b64ndarrays.emplace_back(std::move(blob));
    }
    return g;
  }
};

std::string SaveJSON(const ObjectRef& n) {
  auto jgraph = JSONGraph::Create(n);
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  jgraph.Save(&writer);
  return os.str();
}

// <bojian/TVM-AutoDiff> The upstream implementation only returns the root
//                       operation that is deserialized from the JSON string.
//
//                       However, this creates endless nightmare as it
//                       implicitly instantiates operations such as placeholders
//                       that can neither be connected to previously created
//                       placeholder operations nor be referenced by the Python
//                       frontend. Hence, I changed the interface to return ALL
//                       the tensors deserialized from thee JSON string.
ObjectRef LoadJSON(std::string json_str
    // <bojian/TVM-AutoDiff> Added an extra flag for returning all the tensors.
  , bool ret_all_tensors
    ) {

  // <bojian/TVM-AutoDiff> Added logging when loading the JSON string.
  // LOG(INFO) << "Loading JSON";

  JSONGraph jgraph;
  std::vector<ObjectPtr<Object>> nodes;
  std::vector<runtime::NDArray> tensors;
  {
    // load in json graph.
    std::istringstream is(json_str);
    dmlc::JSONReader reader(&is);
    jgraph.Load(&reader);
    // load in tensors
    for (const std::string& blob : jgraph.b64ndarrays) {
      dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
      support::Base64InStream b64strm(&mstrm);
      b64strm.InitPosition();
      runtime::NDArray temp;
      CHECK(temp.Load(&b64strm));
      tensors.emplace_back(temp);
    }
  }
  ReflectionVTable* reflection = ReflectionVTable::Global();

  // <bojian/TVM-AutoDiff> Use worklist algorithm to map Halide-style CallNode's
  //                       to Tensors (which are generated from OpeartionNode's).
  // create a mapping for looking up all the placeholder nodes
  std::unordered_map<std::string, size_t> ph_name2index_map;
  for (size_t i = 0; i < jgraph.nodes.size(); ++i) {
    const JSONNode& jnode = jgraph.nodes[i];
    if (jnode.type_key == ::tvm::te::PlaceholderOpNode::_type_key) {
      ph_name2index_map[jnode.attrs.at("name")] = i;
    }  // if (jnode.type_key == "PlaceholderOp")
  }  // for (jnode ∈ jgraph.nodes)

  bool jnodes_changed;
  do {
    jnodes_changed = false;
    for (JSONNode& jnode : jgraph.nodes) {
      if (jnode.type_key == ::tvm::tir::CallNode::_type_key) {
        // make a copy of the CallNode's attributes
        const std::string callnode_calltype = jnode.attrs.at("call_type"),
                          callnode_args = jnode.attrs.at("args"),
                          callnode_name = jnode.attrs.at("name"),
                          callnode_dtype = jnode.attrs.at("dtype"),
                          callnode_valueidx = jnode.attrs.at("value_index");

        if (callnode_calltype == "3") {
          LOG(INFO) << "Halide-style CallNode (" << callnode_name
                    << ") encountered. " 
                       "Modifying the nodes for backward compatibility.";
          auto ph_n2imap_iter = ph_name2index_map.find(callnode_name);

          if (ph_n2imap_iter != ph_name2index_map.end()) {
            const size_t& ph_index = ph_n2imap_iter->second;
            const JSONNode& ph_node = jgraph.nodes[ph_index];
            JSONNode tensor_node;

            if (ph_node.attrs.at("dtype") != callnode_dtype) {
              LOG(WARNING) << "Placeholder's data type does not match CallNode's data type: ("
                           << ph_node.attrs.at("dtype") << " != " << callnode_dtype << ").";
            }

            tensor_node.type_key = ::tvm::te::TensorNode::_type_key;
            tensor_node.attrs["op"] = std::to_string(ph_index);
            // copy the shape and data type attributes from the placeholder to the tensor
            tensor_node.attrs["shape"] = ph_node.attrs.at("shape");
            tensor_node.attrs["dtype"] = ph_node.attrs.at("dtype");
            // Assume that the value_index is always 0. Would this be a problem?
            tensor_node.attrs["value_index"] = callnode_valueidx;

            // insert the newly created TensorNode to the end of the jgraph.nodes
            jgraph.nodes.emplace_back(std::move(tensor_node));

            // now, modify the jnode to be "ProducerLoad" node
            jnode.type_key = ::tvm::tir::ProducerLoadNode::_type_key;
            jnode.attrs.clear();
            jnode.attrs["producer"] = std::to_string(jgraph.nodes.size() - 1);
            jnode.attrs["indices"]  = callnode_args;
            jnode.attrs["dtype"]    = callnode_dtype;

            // set the changed flag and break out of the loop
            jnodes_changed = true;
            break;
          } else {  // if (ph_n2imap_iter != ph_name2index_map.end())
            LOG(FATAL) << "Unknown CallNode function name ("
                       << callnode_name << ").";
          }  // if (ph_n2imap_iter != ph_name2index_map.end())
        }  // if (jnode_attr_call_type_iter->second == "3")
      }  // if (jnode.type_key == "Call")
    }  // for (jnode ∈ jgraph.nodes)
  } while (jnodes_changed);

  // node 0 is always null
  nodes.reserve(jgraph.nodes.size());

  for (const JSONNode& jnode : jgraph.nodes) {
    
    // <bojian/TVM-AutoDiff> Added logging on the JSON node type key.
    // LOG(INFO) << "Type Key: " << jnode.type_key;
    // <bojian/TVM-AutoDiff> Added special handing for Halide-style CallNode.
    // if (jnode.type_key == ::tvm::tir::CallNode::_type_key) {
    //   LOG(INFO) << "CallNode has been encountered with repr_bytes="
    //             << jnode.repr_bytes;
    //   for (const std::pair<const std::string, std::string>& kv : jnode.attrs) {
    //     LOG(INFO) << "\t" "attr.k=" << kv.first << ", "
    //                       "attr.v=" << kv.second;
    //   }
    // } 

    if (jnode.type_key == ArrayNode::_type_key) {
      CHECK(jnode.repr_bytes.empty());
      nodes.emplace_back(ArrayNode::CreateRepeated(jnode.data.size(), ObjectRef(nullptr)));
    } else if (jnode.type_key.length() != 0) {
      ObjectPtr<Object> node = reflection->CreateInitObject(jnode.type_key, jnode.repr_bytes);
      nodes.emplace_back(std::move(node));
    } else {
      nodes.emplace_back(ObjectPtr<Object>());
    }
  }
  CHECK_EQ(nodes.size(), jgraph.nodes.size());
  JSONAttrSetter setter;
  setter.node_list_ = &nodes;
  setter.tensor_list_ = &tensors;

  // <bojian/TVM-AutoDiff> Record all the TensorNode's.
  using ::tvm::te::Tensor;
  using ::tvm::te::TensorNode;
  std::vector<size_t> tensor_node_idxs;

  for (size_t i = 0; i < nodes.size(); ++i) {
    setter.node_ = &jgraph.nodes[i];
    // Skip the nodes that has an repr bytes representation.
    // NOTE: the second condition is used to guard the case
    // where the repr bytes itself is an empty string "".
    if (setter.node_->repr_bytes.length() == 0 && nodes[i] != nullptr &&
        !reflection->GetReprBytes(nodes[i].get(), nullptr)) {
      // <bojian/TVM-AutoDiff> Fix for MapNode construction.
      // setter.Set(nodes[i].get());
      setter.Set(&nodes[i]);
    }

    // <bojian/TVM-AutoDiff> Record all the TensorNode's.
    if (nodes[i] != nullptr &&
        nodes[i]->IsInstance<::tvm::te::TensorNode>()) {
      bool tensor_idx_recorded_before = false;
      for (const size_t idx : tensor_node_idxs) {
        if (Tensor(nodes[idx]) == Tensor(nodes[i])) {
          tensor_idx_recorded_before = true;
        }
      }
      if (tensor_idx_recorded_before) {
        continue;
      }
      // LOG(INFO) << "Recording Tensor node " << ObjectRef(nodes[i]);
      tensor_node_idxs.emplace_back(i);
    }

  }

  /* It is challenging to handle cross-node references. Maybe we should consider
     switching to JSON string manipulation?
  // <bojian/TVM-AutoDiff> Use worklist algorithm to map Halide-style CallNode's
  //                       to Tensors (which are generated from OpeartionNode's).
  using ::tvm::te::Operation;
  using ::tvm::te::PlaceholderOpNode;
  using ::tvm::te::TensorNode;
  using ::tvm::tir::CallNode;
  using ::tvm::tir::DataProducer;
  using ::tvm::tir::ProducerLoadNode;

  std::unordered_map<std::string, ObjectPtr<Object>> ph_op_nodes;
  for (ObjectPtr<Object>& node : nodes) {
    if (node == nullptr) continue;
    if (node->IsInstance<PlaceholderOpNode>()) {
      PlaceholderOpNode* pnode = static_cast<PlaceholderOpNode*>(node.get());
      ph_op_nodes[pnode->name] = node;
    }  // if (node->IsInstance<OperationNode>())
  }  // for (node ∈ nodes)

  bool nodes_changed = false;
  do {
    for (ObjectPtr<Object>& node : nodes) {
      if (node == nullptr) continue;
      if (node->IsInstance<CallNode>()) {
        CallNode* pnode = static_cast<CallNode*>(node.get());
        if (pnode->call_type == 3) {
          LOG(INFO) << "Halide-style CallNode (" << pnode->name
                    << ") encountered. " 
                       "Modifying the nodes for backward compatibility.";
          auto ph_op_node = ph_op_nodes.find(pnode->name);
          if (ph_op_node != ph_op_nodes.end()) {
            // LOG(INFO) << "CallNode (" << pnode->name <<  ") has been detected "
            //              "as a PlaceholderNode.";
            ObjectPtr<TensorNode> tensor_node = make_object<TensorNode>();
            ObjectPtr<ProducerLoadNode> producer_load_node
                = make_object<ProducerLoadNode>();
            CHECK(ph_op_node->second->IsInstance<PlaceholderOpNode>());
            PlaceholderOpNode* ph_op_pnode =
                static_cast<PlaceholderOpNode*>(ph_op_node->second.get());

            tensor_node->op = Operation(ph_op_node->second);
            tensor_node->shape = ph_op_pnode->shape;
            tensor_node->dtype = ph_op_pnode->dtype;
            // Assume that the value index is always 0. Would this be a problem?
            tensor_node->value_index = 0;

            producer_load_node->producer = DataProducer(tensor_node);
            producer_load_node->dtype = tensor_node->dtype;
            producer_load_node->indices = pnode->args;

            // reset the CallNode with the newly created ProducerLoadNode
            // node = ;
            // nodes.emplace_back(tensor_node);

            // nodes_changed = true;
            // break;  // set nodes_changed flag to true and break out of the loop
          } else {  if (ph_op_node != ph_op_nodes.end())
            LOG(FATAL) << "Unknown CallNode function name (" << pnode->name << ").";
          }  if (ph_op_node != ph_op_nodes.end())
        }  // if (pnode->call_type == 3)
      }  // if (node->IsInstance<CallNode>())
    }  // for (node ∈ nodes)
  } while (nodes_changed);
   */

  if (!ret_all_tensors || 
      tensor_node_idxs.size() == 0) {
    LOG(INFO) << "Returning the head JSON node "
              << ObjectRef(nodes.at(jgraph.root));
    return ObjectRef(nodes.at(jgraph.root)); 
  }

  std::ostringstream strout;
  strout << "[";
  for (const size_t& idx : tensor_node_idxs) {
    const ObjectPtr<Object> node = nodes[idx];
    TensorNode* pnode = static_cast<TensorNode*>(node.get());
    strout << pnode->op->name << ", ";
  }
  strout << "]";

  LOG(INFO) << "Returning all the TensorNode's recorded: " << strout.str();
  // <bojian/TVM-AutoDiff> Added the logging on the object returned.
  // LOG(INFO) << "Total number of recorded TensorNode's: "
  //           << tensor_node_idxs.size();
  ObjectPtr<ArrayNode> ret
      = ArrayNode::CreateRepeated(tensor_node_idxs.size(), ObjectRef(nullptr));
  for (size_t i = 0; i < tensor_node_idxs.size(); ++i) {
    (*ret)[i] = ObjectRef(nodes[tensor_node_idxs[i]]);
  }

  // <bojian/TVM-AutoDiff> Changed the return value to include all the
  //                       TensorNode's. Please refer to the comment on top of
  //                       LoadJSON for more information.
  // return ObjectRef(nodes.at(jgraph.root));
  return ObjectRef(ret);
}

TVM_REGISTER_GLOBAL("node.SaveJSON").set_body_typed(SaveJSON);

TVM_REGISTER_GLOBAL("node.LoadJSON").set_body_typed(LoadJSON);
}  // namespace tvm
