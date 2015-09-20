// Copyright (c) 2009-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <list>
#include <set>
#include <map>

#include <dp/sg/animation/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Object.h>
#include <dp/util/Reflection.h>

namespace dp
{
  namespace sg
  {
    namespace animation
    {

      /************************************************************************/
      /* Link                                                                 */
      /************************************************************************/
      class Link
      {
      public:
        DP_SG_ANIMATION_API Link(dp::sg::core::ObjectSharedPtr const & srcObject, dp::util::PropertyId srcProperty,
                                 dp::sg::core::ObjectSharedPtr const & dstObject, dp::util::PropertyId dstProperty);

        DP_SG_ANIMATION_API virtual ~Link();

        dp::util::PropertyId  m_srcProperty;
        dp::util::PropertyId  m_dstProperty;
        dp::sg::core::ObjectSharedPtr m_srcObject;
        dp::sg::core::ObjectSharedPtr m_dstObject;

        virtual void sync() const = 0;
      };

      typedef const Link* LinkId;

      /************************************************************************/
      /* LinkInfo                                                             */
      /************************************************************************/
      class LinkInfo
      {
      public:
        DP_SG_ANIMATION_API LinkInfo(dp::sg::core::ObjectSharedPtr const & object) // src reflection object
          : m_reflection(object)
        {
        }

        DP_SG_ANIMATION_API void addLink(LinkId link);
        DP_SG_ANIMATION_API void removeLink(LinkId link);

        bool isTopLevelObject()
        {
          return m_incomingLinks.empty();
        }

        dp::sg::core::ObjectSharedPtr const & getObject()
        {
          return m_reflection;
        }

        bool isEmpty()
        {
          return m_outgoingLinks.empty() && m_incomingLinks.empty();
        }


      protected:
        friend class LinkManager;

        dp::sg::core::ObjectSharedPtr         m_reflection;
        std::list<const Link *>     m_outgoingLinks; // outgoing links
        std::list<const Link *>     m_incomingLinks; // incoming links
      };

      typedef std::map<dp::sg::core::ObjectSharedPtr, LinkInfo> LinkInfoMap;
      typedef std::set<dp::sg::core::ObjectSharedPtr> CalcList;
      typedef std::set<dp::sg::core::ObjectSharedPtr> ReflectionSet;

      struct LinkRuntime;

      /************************************************************************/
      /* LinkManager                                                          */
      /************************************************************************/
      class LinkManager
      {
      public:
        DP_SG_ANIMATION_API LinkManager();
        DP_SG_ANIMATION_API ~LinkManager();

        // link with known type -> runtime check for correct type
        template <typename T> LinkId link( const dp::sg::core::ObjectSharedPtr & srcObject, dp::util::PropertyId srcProperty, const dp::sg::core::ObjectSharedPtr & dstObject, dp::util::PropertyId dstProperty);
        template <typename T> LinkId link( const dp::sg::core::ObjectSharedPtr & srcObject, const char *srcProperty, const dp::sg::core::ObjectSharedPtr & dstObject, const char *dstProperty);

        // link with unknown type -> runtime check for same type on both src and dst
        DP_SG_ANIMATION_API LinkId link( const dp::sg::core::ObjectSharedPtr & srcObject, dp::util::PropertyId srcProperty, const dp::sg::core::ObjectSharedPtr & dstObject, dp::util::PropertyId  dstProperty);
        DP_SG_ANIMATION_API LinkId link( const dp::sg::core::ObjectSharedPtr & srcObject, const char *srcProperty, const dp::sg::core::ObjectSharedPtr & dstObject, const char *dstProperty);

        DP_SG_ANIMATION_API void unlink(LinkId link);

        /** unlink all incoming and outgoing links of this object **/
        DP_SG_ANIMATION_API void unlink( const dp::sg::core::ObjectSharedPtr & object );

        DP_SG_ANIMATION_API bool processLinks();

      protected:

        /** Update processing order of objects if necessary **/
        bool updateLinkOrder();

        void storeLink(LinkId link);

        bool isSourceCalculationDone( const dp::sg::core::ObjectSharedPtr & object );
        bool isGathered( const dp::sg::core::ObjectSharedPtr & object ); // has been added for ordering
        bool isOrdered( const dp::sg::core::ObjectSharedPtr & object ); // has been ordered

        void gather( const dp::sg::core::ObjectSharedPtr & object )
        {
          m_gatheredObjects.insert(object);
        }

        std::list<dp::sg::core::ObjectSharedPtr> getTopLevelObjects();

        LinkInfoMap m_linkInfos;

        ReflectionSet m_gatheredObjects; // objects which have been gathered, but not been place into the ordered list
        ReflectionSet m_orderedObjects; // objects which have been places in the ordered lsit

        std::vector<dp::sg::core::ObjectSharedPtr> m_objectOrder; // objects sorted by dependencies

        bool m_linkOrderDone;
        bool m_cycleDetected;
      };

      template <typename T>
      LinkId LinkManager::link(dp::sg::core::ObjectSharedPtr const & srcObject, dp::util::PropertyId srcProperty,
                               dp::sg::core::ObjectSharedPtr const & dstObject, dp::util::PropertyId  dstProperty)
      {
        DP_ASSERT( srcObjectLock->getPropertyType(srcProperty) == static_cast<Property::Type>(TypedPropertyEnum<T>::type) );
        DP_ASSERT( dstObjectLock->getPropertyType(dstProperty) == static_cast<Property::Type>(TypedPropertyEnum<T>::type) );

        LinkId link = new LinkImpl<T>(srcObjectLock, srcProperty, dstObjectLock, dstProperty);
        storeLink(link);
        return link;
      }

      template <typename T>
      LinkId LinkManager::link(dp::sg::core::ObjectSharedPtr const & srcObject, const char *srcProperty,
                               dp::sg::core::ObjectSharedPtr const & dstObject, const char *dstProperty)
      {
        return link<T>(srcObjectLock, srcObjectLock->getProperty(srcProperty), dstObjectLock, dstObjectLock->getProperty(dstProperty));
      }

    } // namespace animation
  } // namespace sg
} // namespace dp
