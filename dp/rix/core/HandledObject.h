// Copyright NVIDIA Corporation 2011-2012
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

#include <dp/Assert.h>
#include <dp/rix/core/RendererConfig.h>

#include <boost/type_traits.hpp>

#if RIX_CORE_HANDLEDOBJECT_IDS
#include <map>
#endif


namespace dp
{
  namespace rix
  {
    namespace core
    {

      class HandledObject;

      typedef HandledObject* HandledObjectHandle;

      void handleRef( HandledObjectHandle handle );
      void handleUnref( HandledObjectHandle handle );

      template< typename HandleType >
      struct HandleTrait
      {
        typedef HandleType* Type;
      };

      // renderer API classes
      class HandledObject
      {
      public:

      public:
        RIX_CORE_API HandledObject( );

      protected:
        RIX_CORE_API virtual ~HandledObject();

      private:
        RIX_CORE_API virtual void destroy();

        friend void handleRef( HandledObjectHandle handle );
        friend void handleUnref( HandledObjectHandle handle );

      private:
        unsigned int m_refCount;

#if RIX_CORE_HANDLEDOBJECT_IDS
        // debug data for HandledObject instances
        static unsigned int m_nextId;                               // unique id for the next instance
        static std::map< unsigned int, HandledObject* > m_objects;  // map of all instances
        std::map< unsigned int, HandledObject* >* m_ptr;            // ptr to map (to see the map in debugger)
        unsigned int m_id;                                          // id of this instance
#endif
      };

      template<typename HandleType, typename SourceType>
      bool handleIsTypeOf( SourceType handle )
      {
        return !!dynamic_cast<typename HandleTrait<HandleType>::Type>(handle);
      }

      template<typename DestType, typename SourceType>
      typename HandleTrait<DestType>::Type handleCast( SourceType handle )
      {
        DP_ASSERT( !handle || handleIsTypeOf<DestType>( handle ) );
        return static_cast<typename HandleTrait<DestType>::Type>( handle );
      }

      template <typename HandleType>
      void handleReset( HandleType& handle )
      {
        if ( handle )
        {
          handleUnref( handle );
          handle = nullptr;
        }
      }

      inline void handleRef( HandledObjectHandle handle )
      {
        DP_ASSERT( handle );
        ++handle->m_refCount;
      }

      inline void handleUnref( HandledObjectHandle handle )
      {
        DP_ASSERT( handle );
        if ( !--handle->m_refCount )
        {
          handle->destroy();
        }
      }

      template<typename HandleType>
      void handleAssign( HandleType& lhs, const HandleType rhs )
      {
        if ( lhs != rhs)
        {
          if ( rhs )
          {
            handleRef( rhs );
          }

          if ( lhs )
          {
            handleUnref( lhs );
          }

          lhs = rhs;
        }
      }

      template <typename HandleType>
      class SmartHandle
      {
      public:
        SmartHandle()
          : m_handle( nullptr )
        {
        }

        SmartHandle( typename HandleTrait<HandleType>::Type handle )
          : m_handle( handle )
        {
          if ( handle )
          {
            handleRef( handle );
          }
        }

        SmartHandle( const SmartHandle<HandleType> &rhs )
          : m_handle(rhs.m_handle)
        {
          if ( m_handle)
          {
            handleRef( rhs.m_handle );
          }
        }

        template <typename DerivedType>
        SmartHandle( const SmartHandle<DerivedType> &rhs )
          : m_handle(rhs.m_handle)
        {
          DP_STATIC_ASSERT(( boost::is_base_of<HandleType, DerivedType>::value ));
          if ( m_handle)
          {
            handleRef( m_handle );
          }
        }

        ~SmartHandle( )
        {
          if ( m_handle )
          {
            handleUnref( m_handle );
          }
        }

        const SmartHandle<HandleType>& operator=( const SmartHandle<HandleType>& rhs )
        {
          handleAssign( m_handle, rhs.m_handle );
          return *this;
        }

        typename HandleTrait<HandleType>::Type operator->() const
        {
          return m_handle;
        }

        #if _MSC_VER >= 1800
        // From VS2013 on VS supports explicit operators. This fixes all issues with implicit casts for operator bool()
        explicit
        #error "reminder: verify functionality of explicit!"
        #endif
        operator bool() const
        {
          return !!m_handle;
        }

        template< typename RHSHandleType >
        bool operator==( SmartHandle<RHSHandleType> const& rhs )
        {
          DP_STATIC_ASSERT(( boost::is_base_of<HandleType, RHSHandleType>::value | ( boost::is_base_of<RHSHandleType, HandleType>::value )));
          return m_handle == rhs.m_handle;
        }

        template< typename RHSHandleType >
        bool operator!=( SmartHandle<RHSHandleType> const& rhs )
        {
          return !operator==(rhs);
        }

        template< typename RHSHandleType >
        friend bool operator<( SmartHandle<HandleType> const& lhs, SmartHandle<RHSHandleType> const& rhs )
        {
          DP_STATIC_ASSERT(( boost::is_base_of<HandleType, RHSHandleType>::value | ( boost::is_base_of<RHSHandleType, HandleType>::value )));
          return lhs.m_handle < rhs.m_handle;
        }

        typename HandleTrait<HandleType>::Type get() const
        {
          return m_handle;
        }

        template <typename Type>
        Type* get() const
        {
          DP_ASSERT( dynamic_cast<Type*>(m_handle) );
          return static_cast<Type*>(m_handle);
        }

        void reset()
        {
          if ( m_handle )
          {
            handleUnref( m_handle );
            m_handle = nullptr;
          }
        }

      private:
        typename HandleTrait<HandleType>::Type m_handle;
      };

      typedef SmartHandle<HandledObject> SmartHandledObject;

      template<typename HandleType, typename SourceType>
      bool handleIsTypeOf( SmartHandle<SourceType> const & handle )
      {
        return !!dynamic_cast<HandleType*>(handle.get());
      }

      template<typename DestType, typename SourceType>
      SmartHandle< DestType > handleCast( SmartHandle<SourceType> const & handle )
      {
        DP_ASSERT( !handle.get() || handleIsTypeOf<DestType>( handle ) );
        return SmartHandle<DestType>( handleCast<DestType>(handle.get() ) );
      }

      template<typename HandleType>
      void handleAssign( typename HandleTrait<HandleType>::Type & lhs, SmartHandle<HandleType> const & rhs )
      {
        if ( lhs != rhs.get() )
        {
          if ( rhs.get() )
          {
            handleRef( rhs.get() );
          }

          if ( lhs )
          {
            handleUnref( lhs );
          }

          lhs = rhs.get();
        }
      }

    } // namespace core
  } // namespace rix
} // namespace dp

