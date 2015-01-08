// Copyright NVIDIA Corporation 2002-2015
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


#include <dp/sg/core/Object.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/Switch.h>
#include <dp/util/HashGeneratorMurMur.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY(Object, Name);
      DEFINE_STATIC_PROPERTY(Object, Annotation);
      DEFINE_STATIC_PROPERTY(Object, Hints);
      DEFINE_STATIC_PROPERTY(Object, TraversalMask);

      BEGIN_REFLECTION_INFO(Object)
        INIT_STATIC_PROPERTY_RW (Object, Name,          std::string,  SEMANTIC_VALUE, const_reference, const_reference);
        INIT_STATIC_PROPERTY_RW (Object, Annotation,    std::string,  SEMANTIC_VALUE, const_reference, const_reference);
        INIT_STATIC_PROPERTY_RW (Object, Hints,         unsigned int, SEMANTIC_VALUE, value, value);
        INIT_STATIC_PROPERTY_RW (Object, TraversalMask, unsigned int, SEMANTIC_VALUE, value, value);
      END_REFLECTION_INFO

      Object::Object(void)
     : m_objectCode(OC_INVALID)
      , m_hashKey(0)
      , m_name(nullptr)
      , m_annotation(nullptr)
      , m_flags(0)
      , m_hints(0)
      , m_userData(nullptr)
      , m_traversalMask(~0)
      , m_dirtyState(DP_SG_BOUNDING_VOLUMES | DP_SG_HASH_KEY)
      {
      }

      Object::Object(const Object& rhs)
     : m_objectCode(rhs.m_objectCode) // copy object code 
      , m_hashKey(rhs.m_hashKey)
      , m_name(nullptr) // proper pointer initialization
      , m_annotation(nullptr) // dito
      , m_hints(rhs.m_hints)
      , m_flags(rhs.m_flags)
      , m_traversalMask(rhs.m_traversalMask)
      , m_userData(rhs.m_userData) // just copy the address to arbitrary user data
      , m_dirtyState(DP_SG_BOUNDING_VOLUMES | DP_SG_HASH_KEY)
      {
        // concrete objects should have a valid object code - assert this
        DP_ASSERT(m_objectCode!=OC_INVALID);

        if (rhs.m_name)
        { // this copy inherits name from rhs
          m_name = new std::string(*rhs.m_name); 
        }

        if (rhs.m_annotation)
        { // this copy inherits name from rhs
          m_annotation = new std::string(*rhs.m_annotation); 
        }
      }

      Object::~Object(void)
      {
        delete m_name;
        delete m_annotation;
      }

      bool Object::isDataShared(void) const
      {
        return(false);
      }

      DataID Object::getDataID(void) const
      {
        return((DataID)this);
      }

      void Object::setName(const std::string& name)
      {
        if (!m_name || (*m_name != name))
        {
          if (!m_name)
          {
            m_name = new std::string(name);
          }
          else
          {
            *m_name = name;
          }
          notify(PropertyEvent(this, PID_Name));
        }
      }

      const std::string& Object::getName() const
      {
        static std::string noname("");
        return  m_name ? *m_name: noname;
      }

      void Object::setAnnotation(const std::string& anno )
      {
        if (! m_annotation)
        {
          m_annotation = new std::string(anno);
        }
        else
        {
          *m_annotation = anno;
        }
        notify(PropertyEvent(this, PID_Annotation));
      }

      const std::string& Object::getAnnotation() const
      {
        static std::string noanno("");
        return(m_annotation ? *m_annotation: noanno);
      }

      ObjectCode Object::getHigherLevelObjectCode(ObjectCode) const
      {
        return(OC_INVALID);
      }

      Object & Object::operator=(const Object & rhs)
      {
        if (&rhs != this)
        {
          HandledObject::operator=(rhs);

          // concrete objects should have a valid object code - assert this
          DP_ASSERT(m_objectCode != OC_INVALID);
          DP_ASSERT(m_objectCode == rhs.m_objectCode);
      
          // copy mutable data
          m_flags = rhs.m_flags;
          m_hashKey = rhs.m_hashKey;

          if (m_name)
          {
            delete m_name;
            m_name = nullptr;
          }
          if (rhs.m_name)
          { 
            m_name = new std::string(*rhs.m_name); 
          }

          if (m_annotation)
          {
            delete m_annotation;
            m_annotation = nullptr;
          }
          if (rhs.m_annotation)
          {
            m_annotation = new std::string(*rhs.m_annotation);
          }

          // don't copy user data

          // copy hints
          m_hints = rhs.m_hints;

          // copy traversal mask
          m_traversalMask = rhs.m_traversalMask;

          notify(PropertyEvent(this, PID_Annotation));
          notify(PropertyEvent(this, PID_Name));
          notify(PropertyEvent(this, PID_Hints));
        }
        return *this;
      }

      bool Object::isEquivalent(ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare) const
      {
        if (object == this)
        {
          return(true);
        }

        bool equi = (m_objectCode == object->m_objectCode);
        if (!ignoreNames)
        {
          equi = (m_name && object->m_name && (*m_name == *object->m_name)) // objects have the same name
              || (m_name == object->m_name);                                // or both are unnamed
        }
        return(equi);
      }

      util::HashKey Object::getHashKey() const
      {
        if (m_dirtyState & DP_SG_HASH_KEY)
        {
          util::HashGeneratorMurMur hg;
          feedHashGenerator(hg);
          hg.finalize((unsigned int *)&m_hashKey);
          m_dirtyState &= ~DP_SG_HASH_KEY;
        }
        return(m_hashKey);
      }

      unsigned int Object::determineHintsContainment(unsigned int hints) const
      {
        return getHints(hints);
      }

      void Object::feedHashGenerator(util::HashGenerator & hg) const
      {
        // We should always ignore name and annotation!
        hg.update(reinterpret_cast<const unsigned char *>(&m_userData), sizeof(m_userData));  // just take the dumb pointer... that's all we know
        hg.update(reinterpret_cast<const unsigned char *>(&m_hints), sizeof(m_hints));
      }

      void Object::onNotify(dp::util::Event const & event, dp::util::Payload * payload)
      {
        DP_ASSERT(!payload);

        unsigned int changedState = 0;
        switch(event.getType())
        {
          case dp::util::Event::DP_SG_CORE:
            {
              // we can ingore core events
              dp::sg::core::Event const& coreEvent = static_cast<dp::sg::core::Event const&>(event);
              switch(coreEvent.getType())
              {
                case dp::sg::core::Event::GEONODE:
                  {
                    dp::sg::core::GeoNode::Event const& geoNodeEvent = static_cast<dp::sg::core::GeoNode::Event const&>(coreEvent);
                    switch(geoNodeEvent.getType())
                    {
                      case dp::sg::core::GeoNode::Event::PRIMITIVE_CHANGED:
                        changedState |= DP_SG_BOUNDING_VOLUMES;
                        break;
                      default:
                        DP_ASSERT(!"encountered unhandled geonode event type!");
                        break;
                    }
                  }
                  break;
                case dp::sg::core::Event::GROUP:
                  {
                    dp::sg::core::Group::Event const& groupEvent = static_cast<dp::sg::core::Group::Event const&>(coreEvent);
                    switch(groupEvent.getType())
                    {
                      case dp::sg::core::Group::Event::POST_CHILD_ADD:
                      case dp::sg::core::Group::Event::PRE_CHILD_REMOVE:
                      case dp::sg::core::Group::Event::POST_GROUP_EXCHANGED:
                        changedState |= DP_SG_BOUNDING_VOLUMES;
                        break;
                      default:
                        DP_ASSERT(!"encountered unhandled group event type!");
                        break;
                    }
                  }
                  break;
                case dp::sg::core::Event::OBJECT:
                  changedState |= DP_SG_BOUNDING_VOLUMES;
                  break;
                case dp::sg::core::Event::PARAMETER_GROUP_DATA:
                  break;
                default:
                  DP_ASSERT(!"encountered unhandled core event type!");
                  break;
              }
            }
            break;
          case dp::util::Event::GENERIC:
            // no need to change anything on a generic event
            break;
          case dp::util::Event::PROPERTY:
            {
              // the only property event we're interested in is the transparency event
              dp::util::Reflection::PropertyEvent const& propertyEvent = static_cast<dp::util::Reflection::PropertyEvent const&>(event);
              dp::util::PropertyId propertyId = propertyEvent.getPropertyId();
              if ( ( propertyId == dp::sg::core::LOD::PID_Center )
                || ( propertyId == dp::sg::core::Switch::PID_ActiveSwitchMask ) )
              {
                changedState |= DP_SG_BOUNDING_VOLUMES;
              }
#if !defined(NDEBUG)
              else if ( ( propertyId != dp::sg::core::EffectData::PID_Transparent )
                    &&  ( propertyId != dp::sg::core::Object::PID_Name )
                    &&  ( propertyId != dp::sg::core::Object::PID_TraversalMask ) )
              {
                DP_ASSERT(!"encountered unhandled property event type!");
              }
#endif
            }
            break;
          default:
            DP_ASSERT(!"encountered unhandled event type!");
            break;
        }

        if ((m_dirtyState | changedState) != m_dirtyState)
        {
          // we assume, that each and every change that makes this object dirty might also invalidate the hash key
          m_dirtyState |= DP_SG_HASH_KEY;
          notify(event);
        }
      }

      void Object::onDestroyed(dp::util::Subject const & subject, dp::util::Payload* payload)
      {
        DP_ASSERT(false);
      }

      std::string objectCodeToName(ObjectCode oc)
      {
        switch(oc)
        {
          case OC_INVALID:               return("Invalid");
          case OC_SCENE:                 return("Scene");
          case OC_GEONODE:               return("GeoNode");
          case OC_GROUP:                 return("Group");
          case OC_LOD:                   return("LOD");
          case OC_SWITCH:                return("Switch");
          case OC_TRANSFORM:             return("Transform");
          case OC_BILLBOARD:             return("Billboard");
          case OC_CLIPPLANE:             return("ClipPlane");
          case OC_LIGHT_SOURCE:          return("LightSource");
          case OC_VERTEX_ATTRIBUTE_SET:  return("VertexAttributeSet");
          case OC_PRIMITIVE:             return("Primitive");
          case OC_INDEX_SET:             return("IndexSet");
          case OC_EFFECT_DATA:           return("EffectData");
          case OC_PARAMETER_GROUP_DATA:  return("ParameterGroupData");
          case OC_SAMPLER:               return("Sampler");
          case OC_PARALLELCAMERA:        return("ParallelCamera");
          case OC_PERSPECTIVECAMERA:     return("PerspectiveCamera");
          case OC_MATRIXCAMERA:          return("MatrixCamera");
          default:
            std::ostringstream oss;
            oss << "Unknown objectCode: " << oc;
            return(oss.str());
        }
      }

    }//namespace core
  }//namespace sg
}//namespace dp
