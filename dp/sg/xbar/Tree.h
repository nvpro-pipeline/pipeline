// Copyright NVIDIA Corporation 2011-2015
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <deque>
#include <vector>
#include <dp/Assert.h>

namespace dp
{
  namespace sg
  {
    namespace xbar
    {
      typedef dp::Uint32 ObjectTreeIndex;

      template< class NodeClass, class IndexClass >
      class TreeBaseClass
      {
      public:
        // typedefs for traversers
        typedef NodeClass  NodeType;
        typedef IndexClass IndexType;

        TreeBaseClass();
        virtual ~TreeBaseClass();

        IndexClass getFreeNode();

        IndexClass insertNode( const NodeClass & node, IndexClass parentIndex, IndexClass prevSiblingIndex );

        void deleteNode( IndexClass index );

        NodeClass& operator[]( const IndexClass index );

        const NodeClass& operator[]( const IndexClass index ) const;

        size_t size() const;

        void markDirty( const IndexClass index, unsigned int bits );

        IndexClass               m_firstFreeIndex;
        std::vector< NodeClass > m_tree;
        std::vector< IndexClass> m_dirtyObjects;
      };

      template <typename TreeType, typename TreeNodeVisitor>
      class PreOrderTreeTraverser
      {
      public:
        void traverse( TreeType &tree, TreeNodeVisitor &visitor, typename TreeType::IndexType root = 0 );

        void processDirtyList( TreeType &tree, TreeNodeVisitor &visitor, unsigned int dirtyBitMask );

      protected:
        void doTraverse( typename TreeType::IndexType index );

        TreeType        *m_tree;
        TreeNodeVisitor *m_visitor;
      };

      template< class NodeClass, class IndexClass >
      TreeBaseClass<NodeClass, IndexClass>::TreeBaseClass()
      {
        // initialize object tree with some free indices
        IndexType n( 65536 );
        m_tree.resize( n );
        // generate a chain of free objects, leave last index as initialized (~0)
        for( IndexType i=0; i<n-1; ++i )
        {
          m_tree[i].m_nextSibling = i+1;
        }
        m_firstFreeIndex = 0;
      }

      template< class NodeClass, class IndexClass >
      TreeBaseClass<NodeClass, IndexClass>::~TreeBaseClass()
      {
      }

      template< class NodeClass, class IndexClass >
      IndexClass dp::sg::xbar::TreeBaseClass<NodeClass, IndexClass>::getFreeNode()
      {
        // check if this is the last free index -> allocate more
        if( m_tree[m_firstFreeIndex].m_nextSibling == ~0 )
        {
          IndexType size = IndexType(m_tree.size());
          IndexClass firstNew = size;

          // resize to factor of old size
          m_tree.resize( dp::checked_cast<size_t>(size * 1.5f) );
          IndexType newSize = IndexType(m_tree.size());

          // generate a new chain of free objects, with the last one pointing to ~0
          m_tree[m_firstFreeIndex].m_nextSibling = firstNew;
          for( IndexType i = firstNew; i < newSize - 1; ++i )
          {
            m_tree[i].m_nextSibling = i+1;
          }
        }

        IndexType index = m_firstFreeIndex;
        m_firstFreeIndex = m_tree[m_firstFreeIndex].m_nextSibling;
        return index;
      }

      template< class NodeClass, class IndexClass >
      IndexClass TreeBaseClass<NodeClass, IndexClass>::insertNode( const NodeClass & node, IndexClass parentIndex, IndexClass prevSiblingIndex )
      {
        IndexClass index = getFreeNode();

        NodeClass & newNode = m_tree[index];

        // put node into tree
        newNode = node;

        // connect node in tree
        newNode.m_parentIndex = parentIndex;

        if( parentIndex != ~0 )
        {
          NodeClass & parentNode = m_tree[parentIndex];
          // first child below a node
          if ( parentNode.m_firstChild == ~0 )
          {
            parentNode.m_firstChild = index;
          }
          // insert in the beginning of a node
          else if ( prevSiblingIndex == ~0 )
          {
            newNode.m_nextSibling = parentNode.m_firstChild;
            parentNode.m_firstChild = index;
          }
          // insert between two childs or at the end
          else
          {
            newNode.m_nextSibling = m_tree[prevSiblingIndex].m_nextSibling;
            m_tree[prevSiblingIndex].m_nextSibling = index;
          }

          // Use default transform linking
          // Values will be replaced in case of nodes with transforms.
          newNode.m_transform = parentNode.m_transform;
          newNode.m_transformLevel = parentNode.m_transformLevel;
          newNode.m_transformParent = parentNode.m_transformParent;
        }


        markDirty( index, ~0 );

        return index;
      }

      template< class NodeClass, class IndexClass >
      void TreeBaseClass<NodeClass, IndexClass>::deleteNode( IndexClass index )
      {
        DP_ASSERT( index != ~0 );
        NodeClass& node = m_tree[index];

        // sever family ties
        if( node.m_parentIndex != ~0 )
        {
          NodeClass& parent = m_tree[node.m_parentIndex];

          // disconnect node from previous sibling
          IndexClass current = parent.m_firstChild;
          while( current != ~0 )
          {
            if( m_tree[current].m_nextSibling == index )
            {
              // thou art not me brother!
              m_tree[current].m_nextSibling = node.m_nextSibling;
              break;
            }
            current = m_tree[current].m_nextSibling;
          }

          // disconnect node from parent
          if( parent.m_firstChild == index )
          {
            // you are not my child anymore!
            parent.m_firstChild = node.m_nextSibling;
          }

          // I don't have parents!
          node.m_parentIndex = ~0;
        }

        // free node and all nodes below by putting them into the free list
        IndexClass lastIndex = ~0;
        std::deque< ObjectTreeIndex > queue;
        queue.push_back( index );
        while( !queue.empty() )
        {
          // take front node from queue
          IndexClass currentIndex = queue.front();
          NodeClass& current      = m_tree[currentIndex];
          queue.pop_front();

          // insert all children into queue
          IndexClass childIndex = current.m_firstChild;
          while( childIndex != ~0 )
          {
            queue.push_back( childIndex );
            childIndex = m_tree[childIndex].m_nextSibling;
          }

          // disconnect node from parents and children
          current.m_firstChild  = ~0;
          current.m_parentIndex = ~0;

          // put node into free list
          if( lastIndex != ~0 )
          {
            m_tree[lastIndex].m_nextSibling = currentIndex;
          }
          lastIndex = currentIndex;
        }
        m_tree[lastIndex].m_nextSibling = m_firstFreeIndex;
        m_firstFreeIndex = index;
      }

      template< class NodeClass, class IndexClass >
      NodeClass& TreeBaseClass<NodeClass, IndexClass>::operator[]( const IndexClass index )
      {
        return m_tree[index];
      }

      template< class NodeClass, class IndexClass >
      const NodeClass& TreeBaseClass<NodeClass, IndexClass>::operator[]( const IndexClass index ) const
      {
        return m_tree[index];
      }

      template< class NodeClass, class IndexClass >
      size_t TreeBaseClass<NodeClass, IndexClass>::size() const
      {
        return m_tree.size();
      }

      template< class NodeClass, class IndexClass >
      void TreeBaseClass<NodeClass, IndexClass>::markDirty( const IndexClass index, unsigned int bits )
      {
        if ( !m_tree[index].m_dirtyBits )
        {
          m_dirtyObjects.push_back( index );
        }
        m_tree[index].m_dirtyBits |= bits;
      }

      class TreeNodeBaseClass
      {
      public:
        TreeNodeBaseClass()
          : m_parentIndex( ~0 )
          , m_nextSibling( ~0 )
          , m_firstChild( ~0 )
          , m_dirtyBits( 0 )
        {}

        TreeNodeBaseClass( const TreeNodeBaseClass& rhs )
          : m_parentIndex( rhs.m_parentIndex )
          , m_nextSibling( rhs.m_nextSibling )
          , m_firstChild( rhs.m_firstChild )
          , m_dirtyBits( rhs.m_dirtyBits )
        {
        }

        TreeNodeBaseClass& operator=( const TreeNodeBaseClass& rhs )
        {
          m_parentIndex = rhs.m_parentIndex;
          m_nextSibling = rhs.m_nextSibling;
          m_firstChild = rhs.m_firstChild;
          m_dirtyBits = rhs.m_dirtyBits;
          return *this;
        }

      public:
        typedef dp::Uint32 NodeIndex;
        typedef dp::Uint32 DirtyBits;

        NodeIndex m_parentIndex; // index of this node's parent
        NodeIndex m_nextSibling; // index of this node's next sibling (for child iteration)
        NodeIndex m_firstChild;  // index of this node's first child (for child iteration)
        DirtyBits m_dirtyBits;   // vector of dirty bits (bit definition in derived classes)
      };

      template <typename TreeType, typename TreeNodeVisitor>
      void PreOrderTreeTraverser<TreeType, TreeNodeVisitor>::traverse( TreeType &tree, TreeNodeVisitor &visitor, typename TreeType::IndexType root /*= 0 */ )
      {
        m_tree = &tree;
        m_visitor = &visitor;

        doTraverse( root );
      };

      template <typename TreeType, typename TreeNodeVisitor>
      void PreOrderTreeTraverser<TreeType, TreeNodeVisitor>::processDirtyList( TreeType &tree, TreeNodeVisitor &visitor, unsigned int dirtyBitMask )
      {
        m_tree = &tree;
        m_visitor = &visitor;

        typename std::vector<typename TreeType::IndexType>::iterator itEnd = m_tree->m_dirtyObjects.end();
        for ( typename std::vector<typename TreeType::IndexType>::iterator it = m_tree->m_dirtyObjects.begin(); it != itEnd;++it )
        {
          if ( ((*m_tree)[*it].m_dirtyBits & dirtyBitMask) == 0)
          {
            continue;
          }

          // search dirty parent node
          typename TreeType::IndexType dirtyRootNode = ~0;
          typename TreeType::IndexType currentNode = *it;
          while ( currentNode != ~0 )
          {
            if ( ((*m_tree)[currentNode].m_dirtyBits & dirtyBitMask) != 0 )
            {
              dirtyRootNode = currentNode;
            }
            currentNode = (*m_tree)[currentNode].m_parentIndex;
          }

          doTraverse( dirtyRootNode );
        }
      }

      template <typename TreeType, typename TreeNodeVisitor>
      void PreOrderTreeTraverser<TreeType, TreeNodeVisitor>::doTraverse( typename TreeType::IndexType index )
      {
        typename TreeNodeVisitor::Data data;
        if ( m_visitor->preTraverse( index, data ) )
        {
          typename TreeType::IndexType sibling = (*m_tree)[index].m_firstChild;
          while ( sibling != ~0 )
          {
            doTraverse( sibling );
            sibling = (*m_tree)[sibling].m_nextSibling;
          }
        }
        m_visitor->postTraverse( index, data );
      }

    } // namespace xbar
  } // namespace sg
} // namespace dp
