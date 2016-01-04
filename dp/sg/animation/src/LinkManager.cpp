// Copyright (c) 2009-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <iterator>
#include <dp/math/Quatt.h>
#include <dp/math/Trafo.h>
#include <dp/sg/core/VertexAttribute.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/animation/LinkManager.h>

namespace dp
{
  namespace sg
  {
    namespace animation
    {

      /************************************************************************/
      /* Link                                                                 */
      /************************************************************************/
      Link::Link(dp::sg::core::ObjectSharedPtr const & srcObject, dp::util::PropertyId srcProperty,
                 dp::sg::core::ObjectSharedPtr const & dstObject, dp::util::PropertyId dstProperty)
        : m_srcObject(srcObject)
        , m_srcProperty(srcProperty)
        , m_dstObject(dstObject)
        , m_dstProperty(dstProperty)
      {
      }

      Link::~Link()
      {
      }

      /************************************************************************/
      /* LinkImpl                                                             */
      /************************************************************************/
      template <typename T>
      class LinkImpl : public Link
      {
      public:
        LinkImpl(dp::sg::core::ObjectSharedPtr const & srcObject, dp::util::PropertyId srcProperty,
                 dp::sg::core::ObjectSharedPtr const & dstObject, dp::util::PropertyId dstProperty);
        virtual ~LinkImpl();
        virtual void sync() const;
      };

      template <typename T>
      LinkImpl<T>::LinkImpl(dp::sg::core::ObjectSharedPtr const & srcObject, dp::util::PropertyId srcProperty, dp::sg::core::ObjectSharedPtr const & dstObject, dp::util::PropertyId dstProperty)
        : Link(srcObject, srcProperty, dstObject, dstProperty)
      {
      }

      template <typename T>
      LinkImpl<T>::~LinkImpl()
      {
      }

      template <typename T>
      void LinkImpl<T>::sync() const
      {
        m_dstObject->setValue<T>(m_dstProperty, m_srcObject->getValue<T>(m_srcProperty));
      }


      /************/
      /* LinkInfo */
      /************/
      void LinkInfo::addLink(LinkId link)
      {
        if (link->m_srcObject == m_reflection)
        {
          m_outgoingLinks.push_back(link);
        }
        else if (link->m_dstObject == m_reflection)
        {
          m_incomingLinks.push_back(link);
        }
        else
        {
          //std::cout << "m_reflection: " << m_reflection << std::endl;
          DP_ASSERT(0 && "neither incoming nor outgoing reflection object match link info object");
        }
      }

      void LinkInfo::removeLink(LinkId const link)
      {
        if (link->m_srcObject == m_reflection)
        {
          m_outgoingLinks.remove(link);
        }
        else
        {
          m_incomingLinks.remove(link);
        }
      }

      /*****************/
      /** LinkManager **/
      /*****************/
      LinkManager::LinkManager()
        : m_linkOrderDone(false)
      {
      }

      LinkManager::~LinkManager()
      {
        // delete all link objects
        for (LinkInfoMap::iterator it = m_linkInfos.begin(); it != m_linkInfos.end(); ++it)
        {
          for (std::list<const Link *>::iterator itLink = it->second.m_outgoingLinks.begin(); itLink != it->second.m_outgoingLinks.end(); ++itLink)
          {
            delete *itLink;
          }
        }
      }

      /** tore the given link in the link data-structure **/
      void LinkManager::storeLink(LinkId link)
      {
        LinkInfoMap::iterator itSrc = m_linkInfos.find(link->m_srcObject);
        // add new linkinfo
        if (itSrc == m_linkInfos.end())
        {
          std::pair<LinkInfoMap::iterator, bool> result;
          result = m_linkInfos.insert(std::make_pair(link->m_srcObject, LinkInfo(link->m_srcObject)));

          DP_ASSERT(result.second);
          itSrc = result.first;
        }
        itSrc->second.addLink(link);

        LinkInfoMap::iterator itDst = m_linkInfos.find(link->m_dstObject);
        if (itDst == m_linkInfos.end())
        {
          std::pair<LinkInfoMap::iterator, bool> result;
          result = m_linkInfos.insert(std::make_pair(link->m_dstObject, LinkInfo(link->m_dstObject)));

          DP_ASSERT(result.second);
          itDst = result.first;
        }
        itDst->second.addLink(link);
      }

      LinkId LinkManager::link(dp::sg::core::ObjectSharedPtr const & srcObject, dp::util::PropertyId srcProperty,
                               dp::sg::core::ObjectSharedPtr const & dstObject, dp::util::PropertyId dstProperty)
      {
        DP_ASSERT(srcObject->getPropertyType(srcProperty) == dstObject->getPropertyType(dstProperty));

        m_linkOrderDone = false;

        Link *link;
        switch (srcObject->getPropertyType(srcProperty)) {
        case dp::util::Property::Type::FLOAT:
          link = new LinkImpl<float>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::FLOAT2:
          link = new LinkImpl<dp::math::Vec2f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::FLOAT3:
          link = new LinkImpl<dp::math::Vec3f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::FLOAT4:
          link = new LinkImpl<dp::math::Vec4f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::UINT:
          link = new LinkImpl<unsigned int>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::QUATERNION_FLOAT:
          link = new LinkImpl<dp::math::Quatf>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::BOX2_FLOAT:
          link = new LinkImpl<dp::math::Box2f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::BOX3_FLOAT:
          link = new LinkImpl<dp::math::Box3f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::BOX4_FLOAT:
          link = new LinkImpl<dp::math::Box4f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::MATRIX33_FLOAT:
          link = new LinkImpl<dp::math::Mat33f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::MATRIX44_FLOAT:
          link = new LinkImpl<dp::math::Mat44f>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::BOOLEAN:
          link = new LinkImpl<bool>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::TRANSFORMATION:
          link = new LinkImpl<dp::math::Trafo>(srcObject, srcProperty, dstObject, dstProperty);
          break;
        case dp::util::Property::Type::VERTEX_ATTRIBUTE:
          link = new LinkImpl<dp::sg::core::VertexAttribute>(srcObject, srcProperty, dstObject, dstProperty);
          break;

        default:
          link = 0;
        }

        DP_ASSERT(link && "unknown link type");

        if (link)
        {
          storeLink(link);
        }
        return link;
      }


      LinkId LinkManager::link(dp::sg::core::ObjectSharedPtr const & srcObject, const char *srcProperty,
                               dp::sg::core::ObjectSharedPtr const & dstObject, const char *dstProperty)
      {
        return link(srcObject, srcObject->getProperty(srcProperty), dstObject, dstObject->getProperty(dstProperty));
      }

      void LinkManager::unlink(LinkId link)
      {
        if (link)
        {
          LinkInfoMap::iterator itSrc = m_linkInfos.find(link->m_srcObject);
          LinkInfoMap::iterator itDst = m_linkInfos.find(link->m_dstObject);
          if (itSrc != m_linkInfos.end() && itDst != m_linkInfos.end())
          {
            itSrc->second.removeLink(link);
            if (itSrc->second.isEmpty())
            {
              m_linkInfos.erase(itSrc);
            }

            itDst->second.removeLink(link);
            if (itDst->second.isEmpty())
            {
              m_linkInfos.erase(itDst);
            }

            delete link;
          }
          else
          {
            throw std::runtime_error("Invalid link passed to LinkManager::unlink");
          }
        }
      }

      void LinkManager::unlink(dp::sg::core::ObjectSharedPtr const & object)
      {
        LinkInfoMap::iterator it = m_linkInfos.find(object);
        if (it != m_linkInfos.end())
        {
          std::list<LinkId> unlinkList;
          std::copy(it->second.m_incomingLinks.begin(), it->second.m_incomingLinks.end(), std::back_inserter(unlinkList));
          std::copy(it->second.m_outgoingLinks.begin(), it->second.m_outgoingLinks.end(), std::back_inserter(unlinkList));
          for (std::list<LinkId>::const_iterator itLink = unlinkList.begin(); itLink != unlinkList.end(); ++itLink)
          {
            unlink(*itLink);
          }
        }

      }

      std::list<dp::sg::core::ObjectSharedPtr> LinkManager::getTopLevelObjects()
      {
        std::list<dp::sg::core::ObjectSharedPtr> topLevelObjects;

        LinkInfoMap::iterator it = m_linkInfos.begin();
        while (it != m_linkInfos.end())
        {
          if (it->second.isTopLevelObject())
          {
            topLevelObjects.push_back(it->second.getObject());
          }
          ++it;
        }
        return topLevelObjects;
      }

      bool LinkManager::isGathered(dp::sg::core::ObjectSharedPtr const & reflection)
      {
        return (m_gatheredObjects.find(reflection) != m_gatheredObjects.end());
      }

      bool LinkManager::isOrdered(dp::sg::core::ObjectSharedPtr const & reflection)
      {
        return (m_orderedObjects.find(reflection) != m_orderedObjects.end());
      }

      /** returns true if calculation of source reflection objects is done **/
      bool LinkManager::isSourceCalculationDone(dp::sg::core::ObjectSharedPtr const & reflection)
      {
        LinkInfoMap::iterator it = m_linkInfos.find(reflection);
        DP_ASSERT(it != m_linkInfos.end());

        bool ordered = true;
        for (std::list<const Link*>::const_iterator itLink = it->second.m_incomingLinks.begin(); itLink != it->second.m_incomingLinks.end(); ++itLink)
        {
          // source element is not yet calculated
          const Link *link = *itLink;
          if (!isOrdered(link->m_srcObject))
          {
            ordered = false;
            break;
          }
        }
        return ordered;
      }

      // returns true if links have been ordered, false if there was a cycle
      bool LinkManager::updateLinkOrder()
      {
        if (!m_linkOrderDone)
        {
          m_cycleDetected = false;

          std::vector<dp::sg::core::ObjectSharedPtr> objectOrder;
          objectOrder.reserve(m_linkInfos.size());

          // clear existing object order, building up a new up
          // retrieve all objects without inputs. This is the starting wave front
          std::list<dp::sg::core::ObjectSharedPtr> remaining = getTopLevelObjects();

          // if there are no top level objects, but objects linked the full link graph is a cycle.
          if (remaining.empty() && !m_linkInfos.empty())
          {
            m_cycleDetected = true;
          }
          else {
            std::copy(remaining.begin(), remaining.end(), std::inserter(m_gatheredObjects, m_gatheredObjects.begin()));
          }

          // iterate over the remaining objects (wave front) and add all objects whose input is ready

          dp::sg::core::ObjectSharedPtr cycleTest;
          while (!remaining.empty() && !m_cycleDetected)
          {
            dp::sg::core::ObjectSharedPtr reflection = remaining.front();
            remaining.pop_front();

            // check if input objects are already added to ordered list
            if (isSourceCalculationDone(reflection))
            {
              // input dependencies are already in ordered list. This element can be added too
              objectOrder.push_back(reflection);
              m_orderedObjects.insert(reflection);

              // retrieve linkinfo for object
              LinkInfoMap::iterator it = m_linkInfos.find(reflection);
              DP_ASSERT(it != m_linkInfos.end());

              // add all descendants to the new front
              for (std::list<const Link*>::const_iterator itLinks = it->second.m_outgoingLinks.begin(); itLinks != it->second.m_outgoingLinks.end(); ++itLinks)
              {
                const Link *link = *itLinks;
                dp::sg::core::ObjectSharedPtr dstObject = link->m_dstObject;
                if (!isGathered(dstObject))
                {
                  gather(dstObject);
                  remaining.push_back(dstObject);
                }
              }
              // new objects have been added to the working set, reset cycle check
              cycleTest.reset();
            }
            // input dependencies not in ordered list, push back to list
            else
            {
              if (cycleTest == reflection)
              {
                m_cycleDetected = true;
              }
              else {
                remaining.push_back(reflection);
                if (!cycleTest)
                {
                  // if this object is being reached again and no other objects has been added to the working list in between
                  // there is a cycle in the graph
                  cycleTest = reflection;
                }
              }
            }
          }

          // use new object order
          m_objectOrder.swap(objectOrder);

          // clear temporary data structures
          m_gatheredObjects.clear();
          m_orderedObjects.clear();
          m_linkOrderDone = true;

        }
        return !m_cycleDetected;
      }


      // returns false if link ordering could not be calculated because of a cycle in the link graph
      bool LinkManager::processLinks()
      {
        bool linkOrderSuccess = updateLinkOrder();

        if (linkOrderSuccess)
        {
          std::vector<dp::sg::core::ObjectSharedPtr>::const_iterator itEnd = m_objectOrder.end();
          for (std::vector<dp::sg::core::ObjectSharedPtr>::iterator it = m_objectOrder.begin(); it != itEnd; ++it)
          {
            LinkInfoMap::iterator itLinkInfo = m_linkInfos.find(*it);
            std::list<const Link*>::const_iterator itLinkEnd = itLinkInfo->second.m_outgoingLinks.end();
            for (std::list<const Link*>::const_iterator itLink = itLinkInfo->second.m_outgoingLinks.begin(); itLink != itLinkEnd; ++itLink)
            {
              const Link *link = *itLink;
              link->sync();
            }
          }
        }

        return linkOrderSuccess;
      }

    } // namespace animation
  } // namespace sg
} // namespace dp
