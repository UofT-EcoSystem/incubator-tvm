#pragma once


#include "../loop_state.h"
#include "../search_task.h"


namespace tvm {
        namespace ansor {
                namespace symtuning {


class SearchClusterNode : public Object
{
public:
        Array < SearchTask > tasks;
        Array < Array < State > > sketches;
        int repr_idx;

        void VisitAttrs(AttrVisitor * const v)
        {
                v->Visit("tasks", &tasks);
                v->Visit("sketches", &sketches);
                v->Visit("repr_idx", &repr_idx);
        }

        static constexpr const char * _type_key = "ansor.SearchCluster";
        TVM_DECLARE_FINAL_OBJECT_INFO(SearchClusterNode, Object);
};  // class SearchClusterNode


class SearchCluster : public ObjectRef
{
public:
        SearchCluster(Array < SearchTask > tasks,
                      Array < Array < State > > sketches,
                      const int repr_idx);
        TVM_DEFINE_OBJECT_REF_METHODS(SearchCluster, ObjectRef,  
                                      SearchClusterNode);
};  // class SearchCluster


                }  // namespace symtuning
        }  // namespace ansor
}  // namespace tvm
