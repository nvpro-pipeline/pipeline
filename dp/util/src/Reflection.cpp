// Copyright NVIDIA Corporation 2009-2011
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
#include <dp/util/Reflection.h>

namespace dp
{
  namespace util
  {

    Property::~Property()
    {

    }

    Property *Property::clone() const
    {
      return const_cast<Property *>(this);
    }

    PropertyList::~PropertyList()
    {
    }

    void PropertyList::destroy()
    {
      if ( !isStatic() )
      {
        delete this;
      }
    }

    PropertyListImpl::PropertyListImpl( bool isStatic )
      : m_isStatic( isStatic )
    {
    }

    PropertyListImpl::PropertyListImpl( const PropertyListImpl &rhs )
      : PropertyList( rhs )
      , m_isStatic( rhs.m_isStatic )
    {
      for ( PropertyMap::const_iterator it = rhs.m_propertyMap.begin(); it != rhs.m_propertyMap.end(); ++it )
      {
        addProperty( it->first, it->second->clone() );
      }
    }

    PropertyListImpl::~PropertyListImpl()
    {
      // free memory for all property ids
      for (std::vector<PropertyId>::iterator it = m_propertyVector.begin();
         it != m_propertyVector.end();
         ++it)
      {
        (*it)->destroy();
      }
    }

    unsigned int PropertyListImpl::getPropertyCount() const
    {
      return checked_cast<unsigned int>(m_propertyVector.size());
    }

    PropertyId PropertyListImpl::getProperty(unsigned int index) const
    {
      DP_ASSERT( index >= 0 && index < m_propertyVector.size() );

      return m_propertyVector[index];
    }

    PropertyId PropertyListImpl::getProperty(const std::string &name) const
    {
      PropertyMap::const_iterator it = m_propertyMap.find(name);
      if (it != m_propertyMap.end())
      {
        return it->second;
      }
      else 
      {
        return 0;
      }
    }

    std::string PropertyListImpl::getPropertyName(unsigned int index) const
    {
      std::string name;
      PropertyId propertyId = getProperty(index);
      if (propertyId)
      {
        for (PropertyMap::const_iterator it = m_propertyMap.begin();it != m_propertyMap.end();++it)
        {
          if (it->second == propertyId)
          {
            name = it->first;
          }
        }
      }
      return name;
    }

    std::string PropertyListImpl::getPropertyName(const PropertyId propertyId) const
    {
      std::string name;
      for (PropertyMap::const_iterator it = m_propertyMap.begin();it != m_propertyMap.end();++it)
      {
        if (it->second == propertyId)
        {
          name = it->first;
        }
      }
      return name;
    }

    void PropertyListImpl::addProperty(const std::string &name, PropertyId property)
    {
      DP_ASSERT( m_propertyMap.find(name) == m_propertyMap.end());

      m_propertyMap[name] = property;
      m_propertyVector.push_back(property);
    }

    bool PropertyListImpl::hasProperty(const PropertyId propertyId) const
    {
      bool result = false;
      for (PropertyMap::const_iterator it = m_propertyMap.begin();it != m_propertyMap.end();++it)
      {
        if (it->second == propertyId)
        {
          result = true;
          break;
        }
      }
      return result;
    }

    bool PropertyListImpl::hasProperty(const std::string &name) const
    {
      return m_propertyMap.find(name) != m_propertyMap.end();
    }

    bool PropertyListImpl::isStatic() const
    {
      return m_isStatic;
    }

    PropertyList *PropertyListImpl::clone() const
    {
      if ( isStatic() )
      {
        return const_cast<PropertyList *>(static_cast<const PropertyList *>(this));
      }
      return new PropertyListImpl( *this );
    }

    /*********************/
    /* PropertyListChain */
    /*********************/
    PropertyListChain::PropertyListChain( bool isStatic )
      : m_isStatic(isStatic)
    {
    }

    PropertyListChain::PropertyListChain( const PropertyListChain &rhs )
      : PropertyList( rhs )
      , m_isStatic( rhs.m_isStatic )
    {
      for (std::vector<PropertyList*>::const_iterator it = rhs.m_propertyLists.begin(); it != rhs.m_propertyLists.end();++it)
      {
        addPropertyList( (*it)->clone() );
      }
    }

    PropertyListChain::~PropertyListChain()
    {
      // call destroy on sublists
      for (std::vector<PropertyList*>::iterator it = m_propertyLists.begin(); it != m_propertyLists.end();++it)
      {
        (*it)->destroy();
      }
    }

    void PropertyListChain::addPropertyList(PropertyList *propertyList)
    {
      m_propertyLists.push_back(propertyList);
    }

    void PropertyListChain::removePropertyList(PropertyList *propertyList)
    {
      std::vector<PropertyList*>::iterator it = std::find( m_propertyLists.begin(), m_propertyLists.end(), propertyList );
      if ( it != m_propertyLists.end() )
      {
        m_propertyLists.erase( it );
      }
    }

    unsigned int PropertyListChain::getPropertyCount() const
    {
      unsigned int propertyCount = 0;
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); it != m_propertyLists.end();++it)
      {
        PropertyList *list = *it;
        propertyCount += list->getPropertyCount();
      }
      return propertyCount;
    }

    PropertyId PropertyListChain::getProperty(unsigned int index) const
    {
      PropertyId propertyId = 0;
      unsigned int listOffset = 0;
      // search list for the given index
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); !propertyId && it != m_propertyLists.end();++it)
      {
        PropertyList *list = *it;
        unsigned int listIndex = index - listOffset;
        if (listIndex < list->getPropertyCount())
        {
          // found list, retrieve propertyid
          propertyId = list->getProperty(listIndex);
          break;
        }
        // adjust listOffset to fit for next list
        listOffset += list->getPropertyCount();
      }

      if (!propertyId)
      {
        std::cerr << "Warning, invalid property index " << index << std::endl;
      }

      return propertyId;
    }

    PropertyId PropertyListChain::getProperty(const std::string &name) const
    {
      PropertyId propertyId = 0;
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); !propertyId && it != m_propertyLists.end();++it)
      {
        PropertyList *list = *it;
        propertyId = list->getProperty(name);
      }

      // no static property found, search dynamic property
      if (!propertyId)
      {
        // this should be a scenix message not cerr
        //std::cerr << "Warning, property " << name << " not found" << std::endl;
      }

      return propertyId;
    }

    std::string PropertyListChain::getPropertyName(unsigned int index) const
    {
      PropertyId propertyId = 0;
      std::string name;
      unsigned int listOffset = 0;
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); !propertyId && it != m_propertyLists.end();++it)
      {
        PropertyList *list = *it;
        unsigned int listIndex = index - listOffset;
        if (listIndex < list->getPropertyCount())
        {
          name = list->getPropertyName(listIndex);
          break;
        }
        listOffset += list->getPropertyCount();
      }
      return name;
    }

    std::string PropertyListChain::getPropertyName(const PropertyId propertyId) const
    {
      DP_ASSERT( propertyId );

      std::string name;
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); it != m_propertyLists.end();++it)
      {
        PropertyList *list = *it;
        if (list->hasProperty(propertyId))
        {
          name = list->getPropertyName(propertyId);
        }
      }

      return name;
    }

    bool PropertyListChain::hasProperty(const PropertyId propertyId) const
    {
      bool result = false;
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); !result && it != m_propertyLists.end();++it)
      {
        result = (*it)->hasProperty(propertyId);
      }
      return result;
    }

    bool PropertyListChain::hasProperty(const std::string &name) const
    {
      bool result = false;
      for (std::vector<PropertyList*>::const_iterator it = m_propertyLists.begin(); !result && it != m_propertyLists.end();++it)
      {
        result = (*it)->hasProperty(name);
      }
      return result;
    }

    bool PropertyListChain::isStatic() const
    {
      return m_isStatic;
    }

    PropertyList *PropertyListChain::clone() const
    {
      if ( isStatic() )
      {
        return const_cast<PropertyList *>(static_cast<const PropertyList *>(this));
      }
  
      return new PropertyListChain( *this );
    }

    PropertyListChain::PropertyListConstIterator PropertyListChain::beginPropertyLists() const
    {
      return m_propertyLists.begin();
    }

    PropertyListChain::PropertyListConstIterator PropertyListChain::endPropertyLists() const
    {
      return m_propertyLists.end();
    }

    /*******************/
    /* PropertyListImp */
    /*******************/
    Reflection::Reflection( const Reflection &rhs )
    {
      if ( rhs.m_propertyLists )
      {
        m_propertyLists = new PropertyListChain( *rhs.m_propertyLists );
      }
      else
      {
        m_propertyLists = nullptr;
      }
    }

    Reflection::~Reflection()
    {
      delete m_propertyLists;
    }

    const char *Reflection::getClassName() const
    {
      return getReflectionInfo()->getClassName();
    }

    unsigned int Reflection::getPropertyCount() const
    {
      return getReflectionInfo()->getPropertyCount() + (m_propertyLists ? m_propertyLists->getPropertyCount() : 0);
    }

    PropertyId Reflection::getProperty(unsigned int index) const
    {
      ReflectionInfo* staticReflectionInfo = getReflectionInfo();

      unsigned int propertyCount = staticReflectionInfo->getPropertyCount();
      if ( index < propertyCount )
      {
        return staticReflectionInfo->getProperty( index );
      }
      else if ( m_propertyLists )
      {
        return m_propertyLists->getProperty( index - propertyCount );
      }
      else
      {
        return nullptr;
      }
    }

    PropertyId Reflection::getProperty(const std::string &name) const
    {
      PropertyId property = getReflectionInfo()->getProperty( name );
      if ( !property && m_propertyLists )
      {
        property = m_propertyLists->getProperty( name );
      }
      return property;
    }

    std::string Reflection::getPropertyName(unsigned int index) const
    {
      ReflectionInfo* staticReflectionInfo = getReflectionInfo();

      unsigned int propertyCount = staticReflectionInfo->getPropertyCount();
      if ( index < propertyCount )
      {
        return staticReflectionInfo->getPropertyName( index );
      }
      else if ( m_propertyLists )
      {
        return m_propertyLists->getPropertyName( index - propertyCount );
      }
      else
      {
        return std::string();
      }
    }

    std::string Reflection::getPropertyName(const PropertyId propertyId) const
    {
      ReflectionInfo* staticReflectionInfo = getReflectionInfo();

      if ( staticReflectionInfo->hasProperty( propertyId ) )
      {
        return staticReflectionInfo->getPropertyName( propertyId );
      }
      else if ( m_propertyLists )
      {
        return m_propertyLists->getPropertyName( propertyId );
      }
      else
      {
        return std::string();
      }
    }

    bool Reflection::hasProperty( const PropertyId propertyId ) const
    {
      return getReflectionInfo()->hasProperty( propertyId ) || ( m_propertyLists && m_propertyLists->hasProperty( propertyId ) );
    }

    bool Reflection::hasProperty( const std::string &name ) const
    {
      return getReflectionInfo()->hasProperty( name ) || ( m_propertyLists && m_propertyLists->hasProperty( name ) );
    }

    ReflectionInfo* Reflection::getReflectionInfo() const
    {
      return getInternalReflectionInfo();
    }

    ReflectionInfo* Reflection::getInternalReflectionInfo()
    {
      static ReflectionInfo *reflectionInfo = new ReflectionInfo();
      return reflectionInfo;
    }

  } // namespace util
} // namespace dp

