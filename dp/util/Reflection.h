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


#pragma once

#include <map>
#include <set>
#include <vector>
#include <string>
#include <dp/math/Boxnt.h>
#include <dp/math/Quatt.h>
#include <dp/math/Trafo.h>
#include <dp/math/Vecnt.h>
#include <dp/util/Observer.h>
#include <dp/util/Semantic.h>

namespace dp
{
  namespace util
  {

    // forward declarations
    class Reflection;
    class ReflectionInfo;
    class Property;

    /*! \brief Type definition of a PropertyId. */
    typedef Property *PropertyId;

    /** \brief This is the base class for all Properties. It's being used as Handle to a Property. All property implementations need to derive from
        \sa TypedProperty which also adds the get/set methods to the interface.
    **/
    class Property
    {
    public:
      /** Supported property types **/
      enum Type
      {
          TYPE_FLOAT
        , TYPE_FLOAT2
        , TYPE_FLOAT3
        , TYPE_FLOAT4
        , TYPE_INT
        , TYPE_INT2
        , TYPE_INT3
        , TYPE_INT4
        , TYPE_UINT
        , TYPE_UINT2
        , TYPE_UINT3
        , TYPE_UINT4
        , TYPE_QUATERNION_FLOAT
        , TYPE_BOX2_FLOAT
        , TYPE_BOX3_FLOAT
        , TYPE_BOX4_FLOAT
        , TYPE_MATRIX33_FLOAT
        , TYPE_MATRIX44_FLOAT
        , TYPE_BOOLEAN
        , TYPE_CHAR
        , TYPE_TRANSFORMATION
        , TYPE_VERTEX_ATTRIBUTE
        , TYPE_OBJECT     //!< class dp::sg::core::Object
        , TYPE_STRING     //!< class std::string
        , TYPE_SET_UINT   //!< class std::set<unsigned int>
        , TYPE_NODE
        , TYPE_SCENE
        , TYPE_VIEWSTATE
        , TYPE_RENDERTARGET
        , TYPE_TEXTURE
        , TYPE_UNKNOWN
      };

      DP_UTIL_API virtual ~Property();
      DP_UTIL_API virtual Type getType() const = 0;         //!< Returns the type of the Property
      DP_UTIL_API virtual dp::util::Semantic getSemantic() const = 0; //!< Returns the semantic of the Property
      DP_UTIL_API virtual std::string const & getAnnotation() const = 0;  //!< Returns the annotation of a property
      DP_UTIL_API virtual bool isEnum() const = 0;          //!< Returns \c true if the Property is an enum
      DP_UTIL_API virtual std::string const & getEnumTypeName() const = 0;  //!< Returns the name of the enum type as a string
      DP_UTIL_API virtual unsigned int getEnumsCount() const = 0;   //!< Returns the number of enum values, if this an enum; otherwise returns 0
      DP_UTIL_API virtual std::string const & getEnumName( unsigned int idx ) const = 0;    //!< Returns the name of the enum value as a string
      DP_UTIL_API virtual void addRef() = 0;                //!< Increase the refcount of the property
      DP_UTIL_API virtual void destroy() = 0;               //!< Decrease the refcount and destroy the property if necessary  
      DP_UTIL_API virtual Property *clone() const; //!< Clone a property. Most properties do not contain any data and can just be reused.
    };

    /*! \brief Helper cast operator for reflection types.
     *  \param object pointer to cast form \a InputType to \a ReturnType. */
  #ifndef NDEBUG
    template <typename ReturnType, typename InputType> ReturnType* reflection_cast(InputType *object)
    {
      ReturnType *castedObject = dynamic_cast<ReturnType*>(object);
      DP_ASSERT(castedObject != 0);
      return castedObject;
    }
  #else
    template <typename ReturnType, typename InputType> ReturnType* reflection_cast(InputType *object)
    {
      return static_cast<ReturnType*>(object);
    }

  #endif

    /* \brief Template trait to get the property value for a C++ datatype 
       \remarks Usage: TypedPropertyEnum<DataType>::Type
    */
    template <typename T> struct TypedPropertyEnum;

    /*! \brief Specialization of the TypedPropertyEnum template for type float. */
    template <> struct TypedPropertyEnum<float> {
      enum { type = Property::TYPE_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec2f. */
    template <> struct TypedPropertyEnum<dp::math::Vec2f> {
      enum { type = Property::TYPE_FLOAT2 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec3f. */
    template <> struct TypedPropertyEnum<dp::math::Vec3f> {
      enum { type = Property::TYPE_FLOAT3 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec4f. */
    template <> struct TypedPropertyEnum<dp::math::Vec4f> {
      enum { type = Property::TYPE_FLOAT4 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Box2f. */
    template <> struct TypedPropertyEnum<dp::math::Box2f> {
      enum { type = Property::TYPE_BOX2_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Box3f. */
    template <> struct TypedPropertyEnum<dp::math::Box3f> {
      enum { type = Property::TYPE_BOX3_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Box4f. */
    template <> struct TypedPropertyEnum<dp::math::Box4f> {
      enum { type = Property::TYPE_BOX4_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Mat33f. */
    template <> struct TypedPropertyEnum<dp::math::Mat33f> {
      enum { type = Property::TYPE_MATRIX33_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Mat44f. */
    template <> struct TypedPropertyEnum<dp::math::Mat44f> {
      enum { type = Property::TYPE_MATRIX44_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Quatf. */
    template <> struct TypedPropertyEnum<dp::math::Quatf> {
      enum { type = Property::TYPE_QUATERNION_FLOAT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type bool. */
    template <> struct TypedPropertyEnum<bool> {
      enum { type = Property::TYPE_BOOLEAN };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type char. */
    template <> struct TypedPropertyEnum<char> {
      enum { type = Property::TYPE_CHAR };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type unsigned int. */
    template <> struct TypedPropertyEnum<unsigned int> {
      enum { type = Property::TYPE_UINT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec2ui. */
    template <> struct TypedPropertyEnum<dp::math::Vec2ui> {
      enum { type = Property::TYPE_UINT2 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec3ui. */
    template <> struct TypedPropertyEnum<dp::math::Vec3ui> {
      enum { type = Property::TYPE_UINT3 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec4ui. */
    template <> struct TypedPropertyEnum<dp::math::Vec4ui> {
      enum { type = Property::TYPE_UINT4 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type int. */
    template <> struct TypedPropertyEnum<int> {
      enum { type = Property::TYPE_INT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec2i. */
    template <> struct TypedPropertyEnum<dp::math::Vec2i> {
      enum { type = Property::TYPE_INT2 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec3i. */
    template <> struct TypedPropertyEnum<dp::math::Vec3i> {
      enum { type = Property::TYPE_INT3 };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Vec4i. */
    template <> struct TypedPropertyEnum<dp::math::Vec4i> {
      enum { type = Property::TYPE_INT4 };
    };
  
    /*! \brief Specialization of the TypedPropertyEnum template for type dp::math::Trafo. */
    template <> struct TypedPropertyEnum<dp::math::Trafo> {
      enum { type = Property::TYPE_TRANSFORMATION };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type Reflection. */
    template <> struct TypedPropertyEnum<Reflection> {
      enum { type = Property::TYPE_OBJECT };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type std::string. */
    template <> struct TypedPropertyEnum<std::string> {
      enum { type = Property::TYPE_STRING };
    };

    /*! \brief Specialization of the TypedPropertyEnum template for type std::set<unsigned int>. */
    template <> struct TypedPropertyEnum<std::set<unsigned int> > {
      enum { type = Property::TYPE_SET_UINT };
    };

    // TODO must be moved to new dp::(ui?) module
  #if 0
    /*! \brief Specialization of the TypedPropertyEnum template for type dp::ui::RenderTargetSharedPtr. */
    template <> struct TypedPropertyEnum< dp::ui::RenderTargetSharedPtr > {
      enum { type = Property::TYPE_RENDERTARGET };
    };
  #endif

    /*! \brief This is the templated baseclass for all properties of a given type<T>. All properties must
     *         derive from this class
     */
    template <typename T> class TypedProperty : public Property
    {
    public:
      virtual ~TypedProperty() {}

      /*! \brief Retrieve the value of a property
       *  \param owner Object to retrieve the property value from
       *  \param value The property value will the stored in this variable
      */
      virtual void getValue(const Reflection *owner, T &value) const = 0;

      /*! \brief Set the value of a property
       *  \param owner Object which will receive the new property value
       *  \param value The new property value which should be set
      */
      virtual void setValue(Reflection *owner, const T&value) = 0;

      /* \brief returns the type of a property. This function uses the \sa TypedPropertyEnum<T> trait. When adding
                new types it's necessary to specialize TypedPropertyEnum<MyNewType>
      */
      virtual Type getType() const { return static_cast<Type>(TypedPropertyEnum<T>::type); }
    };

    /** \brief Type for the TypeTraits template **/
    struct const_reference;
    /** \brief Type for the TypeTraits template **/
    struct value;

    /** \brief This template convertes a pair ValueType/(const_reference/value) to the corresponding parameter signature.
               TypeTraits<ValueType, value>::value_type is of type ValueType
               TypeTraits<ValueType, value>::parameter_type is of type ValueType
               TypeTraits<ValueType, const_reference>::value_type is of type ValueType
               TypeTraits<ValueType, const_reference>::parameter_type is of type const ValueType &
    **/
    template <typename ValueType, typename ParameterType>
    struct  TypeTraits;

    /*! \brief Partial specialization of the TypeTraits template for value types. */
    template <typename ValueType>
    struct TypeTraits<ValueType, value>
    {
        typedef ValueType value_type;       //!< Value type definition of the template parameter \a ValueType
        typedef ValueType parameter_type;   //!< Parameter type definition of the template parameter \a ValueType
    };
  
    /*! \brief Partial specialization of the TypeTraits template for const reference types. */
    template <typename ValueType>
    struct TypeTraits<ValueType, const_reference>
    {
        typedef ValueType        value_type;      //!< Value type definition of the template parameter \a ValueType
        typedef const ValueType& parameter_type;  //!< Parameter type definition of the template parameter \a ValueType
    };

    /** \brief Functor to set a property value in an object
        \param ValueType The value type of the property
        \param SetType The value type accepted by set function used to set the property
        \param ObjectType The class the set function belongs to
        \param set Function pointer which is being called to set the property
    **/
    // FIXME ObjectType not necessary. It should be possible to use Reflection::*set for all function pointers.
    template <typename ValueType, typename SetType, typename ObjectType, void(ObjectType::*set)(SetType)>
    class FunctorSet
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the function to call.
       *  \param value The value to set. */
      void operator()(Reflection *owner, const ValueType &value)
      {
        (reflection_cast<ObjectType>(owner)->*set)(value);
      }
    };

    /*! \brief Functor to set the value of a member of an object.
     *  \param ValueType The value type of the member.
     *  \param ObjectType The class the member belongs to.
     *  \param Member A pointer to the member to set the value to.
     *  \param readonly bool specifying if the member is to be handled as readonly. */
    template <typename ValueType, typename ObjectType, ValueType ObjectType::*Member, bool readonly>
    class FunctorSetMember
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the member to set.
       *  \param value The value to set. */
      void operator()(Reflection *owner, const ValueType &value)
      {
        if (readonly)
        {
          DP_ASSERT( 0 && "set invalid on this property" );
        }
        else
        {
          static_cast<ObjectType*>(owner)->*Member = value;
        }
      }
    };

    /** \brief Functor to set an enum property value in an object. This is special because enums can be either set as ints or by their enum type.
        \param ValueType The value type of the property
        \param SetType The value type accepted by set function used to set the property
        \param ObjectType The class the set function belongs to
        \param set Function pointer which is being called to set the property
    **/
    template <typename ValueType, typename SetType, typename ObjectType, void(ObjectType::*set)(SetType)>
    class FunctorSetEnum
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the function to call.
       *  \param value The value to set. */
      void operator()(Reflection *owner, const ValueType &value)
      {
        (reflection_cast<ObjectType>(owner)->*set)(value);
      }

      /*! \brief Function call operator of this Functor, specialized for values of type int.
       *  \param owner A pointer to the object owning the function to call.
       *  \param value The value to set. */
      void operator()(Reflection *owner, const int &value)
      {
        (reflection_cast<ObjectType>(owner)->*set)( static_cast<ValueType>(value) );
      }
    };

    /** \brief Functor to set a value which always asserts. This is being used for RO properties.
        \param ValueType The value type of the property
        \param ObjectType The class the set function belongs to
    **/
    // FIXME ObjectType unused
    template <typename ValueType, typename ObjectType>
    class FunctorSetInvalid
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object to use.
       *  \param value The value to set. */
      void operator()(const Reflection *owner, const ValueType &value)
      {
        DP_ASSERT( 0 && "set invalid on this property" );
      }

    };

    /** \brief Functor to get a property value from an object
        \param ValueType The value type of the property
        \param ResultType The value type accepted by set function used to set the property
        \param ObjectType The class the get function belongs to
        \param set Function pointer which is being called to get the property
    **/
    template <typename ValueType, typename ResultType, typename ObjectType, ResultType(ObjectType::*get)() const>
    class FunctorGet
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the function to call.
       *  \param returnValue Reference to the memory to get the result. */
      void operator()(const Reflection *owner, ValueType &returnValue)
      {
        returnValue = (reflection_cast<const ObjectType>(owner)->*get)();
      }
    };

    /*! \brief Functor to get a member value from an object.
     *  \param ValueType The value type of the member.
     *  \param ObjectType The class the member belongs to.
     *  \param member A pointer to the member to get. */
    template <typename ValueType, typename ObjectType, ValueType ObjectType::*member>
    class FunctorGetMember
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the member to get.
       *  \param returnValue Reference to the memory to get the member value. */
      void operator()(const Reflection *owner, ValueType &returnValue)
      {
        returnValue = static_cast<const ObjectType*>(owner)->*member;
      }
    };


    /** \brief Functor to get an enum property value from an object. It's possible to use an int or enum as return value.
        \param ValueType The value type of the property
        \param ResultType The value type accepted by set function used to set the property
        \param ObjectType The class the get function belongs to
        \param set Function pointer which is being called to get the property
    **/
    template <typename ValueType, typename ResultType, typename ObjectType, ResultType(ObjectType::*get)() const>
    class FunctorGetEnum
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the function to call.
       *  \param returnValue Reference to the memory to get the function return value. */
      void operator()(const Reflection *owner, ValueType &returnValue)
      {
        returnValue = (reflection_cast<const ObjectType>(owner)->*get)();
      }

      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the function to call.
       *  \param returnValue Reference to the memory to get the function return value. */
      void operator()(const Reflection *owner, int &returnValue)
      {
        returnValue = static_cast<int>( (reflection_cast<const ObjectType>(owner)->*get)() );
      }
    };


    /** \brief Functor to set a value which always asserts. This is being used for RO properties.
        \param ValueType The value type of the property
    **/
    // FIXME ObjectType unused
    // FIXME get unused
    template <typename ValueType, typename ObjectType, const ValueType&(ObjectType::*get)() const>
    class FunctorGetInvalid
    {
    public:
      /*! \brief Function call operator of this Functor
       *  \param owner A pointer to the object owning the function to call.
       *  \param returnValue Reference to the memory to get the function return value. */
      void operator()( const Reflection *owner, ValueType &returnValue )
      {
        DP_ASSERT( 0 && "get invalid on this property" );
      }
    };


    /** \brief A delegating property implementation. This class is used for most of the properties.
        \param ValueType The value type of the Property
        \param ObjectType The Object class this property belongs to
        \param FunctorGet A functor class which must support the following interface <code>void operator()(Reflection *owner, const ValueType &value)</code>
        \param FunctorGet A functor class which must support the following interface <code>void operator()(const Reflection *owner, ValueType &returnValue)</code>
        \param semantic The semantic of this property.
     * corresponding get/set functions of the given class.
     */
    template <typename ValueType, typename ObjectType, typename FunctorGet, typename FunctorSet, typename dp::util::Semantic semantic, bool deleteOnDestroy>
    class TypedPropertyImpl : public TypedProperty<ValueType>
    {
    public:
      TypedPropertyImpl() 
        : m_refCount(1)
      {}
      virtual ~TypedPropertyImpl() {}

      virtual void getValue(const Reflection *owner, ValueType &value) const
      {
        FunctorGet g;
        return g(owner, value);
      }

      virtual void setValue(Reflection *owner, const ValueType &value)
      {
        FunctorSet s;
        s(owner, value);
      }

      //!\brief Annotations are not supported for this object
      virtual std::string const & getAnnotation() const
      {
        static std::string emptyString;
        return( emptyString );
      }

      virtual dp::util::Semantic getSemantic() const
      {
        return semantic;
      }

      virtual bool isEnum() const
      {
        return( false );
      }

      virtual std::string const & getEnumTypeName() const
      {
        static std::string emptyString;
        return( emptyString );
      }

      virtual unsigned int getEnumsCount() const
      {
        return( 0 );
      }

      virtual std::string const & getEnumName( unsigned int idx ) const
      {
        static std::string emptyString;
        return( emptyString );
      }

      virtual void addRef()
      {
        ++m_refCount;
      }

      virtual void destroy()
      {
        if ( deleteOnDestroy )
        {
          --m_refCount;
          if ( !m_refCount )
          {
            delete this;
          }
        }
      }
    protected:
      unsigned int m_refCount;
    };

    /** \brief A property object holding a value directly. 
        \param ValueType The value type the Property should hold.
     *  \remarks This property object should be hold only by a single owner unless different 
     *           holders should hold the same property value.
     */
    template <typename ValueType>
    class TypedPropertyValue : public TypedProperty<ValueType>
    {
    public:
      /*! \brief Constructor of a TypedPropertyValue.
       *  \param semantic The semantic of the property.
       *  \param annotation The annotation of the property.
       *  \param destroy If \c true, this property is deleted on calling destroy.
       *  \param value The initial value of this property. */
      TypedPropertyValue( dp::util::Semantic semantic, const std::string &annotation, bool destroy, const ValueType &value );

      /*! \brief Copy construtor of a TypedPropertyValue.
       *  \param rhs The TypedPropertyValue to copy from. */
      TypedPropertyValue( const TypedPropertyValue &rhs );

      /*! \brief Get the value of this TypedPropertyValue.
       *  \param value A reference to a ValueType to get the value to. */
      virtual void getValue(const Reflection *, ValueType &value) const;

      /*! \brief Set the value of this TypedPropertyValue.
       *  \param value A reference to the value to set. */
      virtual void setValue( Reflection *, const ValueType &value );

      virtual Property::Type getType() const;              //!< Returns the type of the property
      virtual dp::util::Semantic getSemantic() const;      //!< Returns the semantic of the property
      virtual void addRef();                               //!< Not supported on this class
      virtual void destroy();                              //!< Destroy the property if necessary
      virtual std::string getAnnotation() const;           //!< Returns the annotation of the property
      virtual Property *clone() const;                     //!< Return a new Property with the same values.
    private:
      dp::util::Semantic m_semantic;
      const std::string  m_annotation;
      ValueType          m_value;
      bool               m_destroy;
    };

    template <typename ValueType>
    TypedPropertyValue<ValueType>::TypedPropertyValue( dp::util::Semantic semantic, const std::string &annotation, bool destroy, const ValueType &value )
      : m_semantic(semantic)
      , m_annotation(annotation)
      , m_destroy(destroy)
      , m_value(value)
    {
    }

    template <typename ValueType>
    TypedPropertyValue<ValueType>::TypedPropertyValue( const TypedPropertyValue<ValueType> &rhs )
      : m_semantic( rhs.m_semantic )
      , m_annotation( rhs.m_annotation )
      , m_destroy( rhs.m_destroy )
      , m_value( rhs.m_value )
    {
    }

    template <typename ValueType>
    Property::Type TypedPropertyValue<ValueType>::getType() const
    {
      return static_cast<Property::Type>(TypedPropertyEnum<ValueType>::type);
    }

    template <typename ValueType>
    dp::util::Semantic TypedPropertyValue<ValueType>::getSemantic() const
    {
      return m_semantic;
    }

    template <typename ValueType>
    void TypedPropertyValue<ValueType>::addRef()
    {
      DP_ASSERT( 0 && "addRef not supported on TypedPropertyValue");
    }

    template <typename ValueType>
    void TypedPropertyValue<ValueType>::destroy()
    {
      if ( m_destroy )
      {
        delete this;
      }
    }

    template <typename ValueType>
    void TypedPropertyValue<ValueType>::getValue(const Reflection *, ValueType &value ) const
    {
      value = m_value;
    }

    template <typename ValueType>
    void TypedPropertyValue<ValueType>::setValue( Reflection *, const ValueType &value )
    {
      m_value = value;
    }

    template <typename ValueType>
    std::string TypedPropertyValue<ValueType>::getAnnotation( ) const
    {
      return m_annotation;
    }

    template <typename ValueType>
    PropertyId TypedPropertyValue<ValueType>::clone() const
    {
      return new TypedPropertyValue<ValueType>( *this );
    }


    template<typename T>
    struct EnumReflection
    {
      static const std::string                        name;
      static const std::map<unsigned int,std::string> values;
    };

    /** \brief A delegating property implementation with enum as int support. This class is used for all enum properties.
        \param ValueType The value type of the Property
        \param ObjectType The Object class this property belongs to
        \param FunctorGet A functor class which must support the following interface <code>void operator()(Reflection *owner, const ValueType &value)</code>
        \param FunctorGet A functor class which must support the following interface <code>void operator()(const Reflection *owner, ValueType &returnValue)</code>
        \param semantic The semantic of this property.
     * corresponding get/set functions of the given class.
     */
    //TODO implement getEnumTypeName
    template <typename ValueType, typename ObjectType, typename FunctorGet, typename FunctorSet, typename dp::util::Semantic semantic, bool deleteOnDestroy>
    class TypedPropertyImplEnum : public TypedPropertyImpl<int, ObjectType, FunctorGet, FunctorSet, semantic, deleteOnDestroy>
    {
    public:
      TypedPropertyImplEnum() {}
      virtual ~TypedPropertyImplEnum() {}

      /*! \brief Get the value of this TypedPropertyImplEnum.
       *  \param owner The object to get the value from.
       *  \param value Reference of the ValueType to get the value. */
      virtual void getValue(const Reflection *owner, ValueType &value) const
      {
        FunctorGet g;
        g(owner, value);
      }

      /*! \brief Set the value of this TypedPropertyImplEnum.
       *  \param owner The object to set the value in.
       *  \param value The value to set. */
      virtual void setValue(Reflection *owner, const ValueType &value)
      {
        FunctorSet s;
        s(owner, value);
      }

      virtual bool isEnum() const
      {
        DP_ASSERT( std::is_enum<ValueType>::value );
        return( true );
      }

      virtual std::string const & getEnumTypeName() const
      {
        return( EnumReflection<ValueType>::name );
      }

      virtual unsigned int getEnumsCount() const
      {
        return( dp::util::checked_cast<unsigned int>(EnumReflection<ValueType>::values.size()) );
      }

      virtual std::string const & getEnumName( unsigned int idx ) const
      {
        static std::string emptyString;
        std::map<unsigned int,std::string>::const_iterator it = EnumReflection<ValueType>::values.find( idx );
        return( ( it != EnumReflection<ValueType>::values.end() ) ? it->second : emptyString );
      }

    };

    /** \brief This is the base class for a list of Properties. It's possible to implement this interace for custom property lists, i.e. mapping
               properties of an CgFx or an RTFx.
    **/
    class PropertyList {
    public:
      /*! \brief Default constructor of a PropertyList. */
      PropertyList() : m_id(0) {}

      /*! \brief Copy constructor of a PropertyList.
       *  \param rhs The PropertyList to copy from. */
      PropertyList( const PropertyList &rhs ) { m_id = rhs.m_id; }

      /*! \brief Destructor of a PropertyList. */
      DP_UTIL_API virtual ~PropertyList();

      /*! \brief Retrieve the numer of properties in this list.
       *  \return Number of properties within this list
      **/
      DP_UTIL_API virtual unsigned int getPropertyCount() const = 0;

      /*! \brief Get a PropertyId by index. Note that indices may change during runtime if an object has dynamic properties. Never keep indices to properties!
       *  \param index index of Property to get
       *  \return PropertyId of Property at position index */
      DP_UTIL_API virtual PropertyId getProperty(unsigned int index) const = 0;

      /*! \brief Get a PropertyId by name.
       *  \param name name of Property to get
       *  \return PropertyId of Property with given name */
      DP_UTIL_API virtual PropertyId getProperty(const std::string &name) const = 0;

      /*! \brief Get the name of a property at the given index
       *  \param index index of a property
       *  \return std::string with the name of the property at the given index */
      DP_UTIL_API virtual std::string getPropertyName(unsigned int index) const = 0;

      /*! \brief Get the name of a property for the given propertyid
       *  \param propertyId PropertyId of the property which name is requested
       *  \return std::string with the name of the property for the given PropertyId */
      DP_UTIL_API virtual std::string getPropertyName(const PropertyId propertyId) const = 0;

      /*! \brief Check if a given Property is still available
       *  \param propertyId The PropertyId to check for
       *  \return true if the object contains the given Property, false otherwise */
      DP_UTIL_API virtual bool hasProperty(const PropertyId propertyId) const = 0;

      /*! \brief Check if a given Property is still available
       *  \param name name of Property to check for
       *  \return true if the object contains the given Property, false otherwise */
      DP_UTIL_API virtual bool hasProperty(const std::string &name) const = 0;

      /*! \brief Create a deep copy of this PropertyList. */
      DP_UTIL_API virtual PropertyList *clone() const = 0;

      /*! \brief Check if list is marked as static
       *  \return true If list can be threated as static list. Static lists contain only static (program global)
       *               properties and the content of the list is also static after first initialization. */
      DP_UTIL_API virtual bool isStatic() const = 0;

      /** \brief Destroy the given list **/
      DP_UTIL_API virtual void destroy();

      /** \brief Returns id of the list **/
      size_t getId() const { return m_id; }

      /** \brief Sets id for the list
          \param Id new Id for the list
      **/
      void setId( size_t Id) { m_id = Id; }
    private:
      size_t m_id;
    };

    /** \brief Default implementation of a PropertyList. Add the new function addProperty to add Properties
    */
    class PropertyListImpl : public PropertyList
    {
    public:
      /** \brief Constructor for a list of PropertyLists.
          \param isStatic Marks the list as global static or object centric local list.
      **/
      DP_UTIL_API PropertyListImpl( bool isStatic = true );

      /*! \brief Copy constructor of a PropertyListImpl.
       *  \param rhs The PropertyListImpl to copy from. */
      DP_UTIL_API PropertyListImpl( const PropertyListImpl &rhs );

      /*! \brief Destructor of a PropertyListImpl. */
      DP_UTIL_API ~PropertyListImpl( );

      /*! \brief Get the number of properties in this PropertyListImpl.
       *  \returns The number of properties in this PropertyListImpl. */
      DP_UTIL_API unsigned int getPropertyCount() const;

      /*! \brief Get the PropertyId by its index.
       *  \param index The index of the property to handle.
       *  \returns The PropertyId of the property at position \a index. */
      DP_UTIL_API PropertyId getProperty(unsigned int index) const;

      /*! \brief Get the PropertyId by its name.
       *  \param name The name of the property to handle.
       *  \returns The Propertyid of the property named \a name. */
      DP_UTIL_API PropertyId getProperty(const std::string &name) const;

      /** \brief Add a new Property to the PropertyList.
          \param name The name of the property
          \param property The PropertyId of the property (which is a pointer to the property)
      **/
      DP_UTIL_API void addProperty(const std::string &name, PropertyId property);

      /*! \brief Get the property name by its index.
       *  \param index The index of the property to handle.
       *  \returns The name of the property at position \a index. */
      DP_UTIL_API std::string getPropertyName(unsigned int index) const;

      /*! \brief Get the property name by its PropertyId.
       *  \param propertyId The PropertyId of the property to handle.
       *  \returns The name of the property with PropertyId \a propertyId. */
      DP_UTIL_API std::string getPropertyName(const PropertyId propertyId) const;

      /*! \brief Check if this PropertyListImpl has a property specified by a PropertyId.
       *  \param propertyId The PropertyId of the property to check.
       *  \returns \c true if the property with the PropertyId \a propertyId is part of this PropertyListImpl,
       *  otherwise \c false.      */
      DP_UTIL_API virtual bool hasProperty(const PropertyId propertyId) const;

      /*! \brief Check if this PropertyListImpl has a property specified by a name.
       *  \param name The name of the property to check.
       *  \returns \c true if a property named \a name is part of this PropertyListImpl, otherwise \c false. */
      DP_UTIL_API virtual bool hasProperty(const std::string &name) const;

      /*! \brief Create a clone of this PropertyListImpl.
       *  \returns A pointer to the newly created PropertyList. */
      DP_UTIL_API virtual PropertyList *clone() const;

      /*! \brief Check if this PropertyListImpl is static.
       *  \returns \c true if this PropertyListImple is static, otherwise \c false. */
      DP_UTIL_API virtual bool isStatic() const;

    protected:
      typedef std::vector<PropertyId> PropertyVector;         //!< Type definition for container holding properties
      typedef std::map<std::string, PropertyId> PropertyMap;  //!< Type definition for container mapping property names to property

      bool             m_isStatic;          //!< Flag specifying if this PropertyList is static
      PropertyMap      m_propertyMap;       //!< The map from property name to property
      PropertyVector   m_propertyVector;    //!< The container of properties
    };

    /** \brief This PropertyList implementation will provide a view on multiple PropertyList with a single List only.
               Changes to the added PropertyLists are immideatly visible in this view.
    **/
    class PropertyListChain : public PropertyList
    {

    public:
      /*! \brief The container type of the Camera's head lights */
      typedef std::vector<PropertyList*>                                             PropertyListContainer;

      /*! \brief The iterator over the HeadLightContainer */
      typedef PropertyListContainer::iterator PropertyListIterator;

      /*! \brief The const iterator over the HeadLightContainer */
      typedef PropertyListContainer::const_iterator PropertyListConstIterator;

      /** \brief Constructor for a list of PropertyLists.
          \param isStatic Marks the list as global static or object centric local list.
      **/
      DP_UTIL_API PropertyListChain( bool isStatic = true );

      /*! \brief Copy constructor of a PropertyListChain
       *  \param rhs The PropertyListChain to copy from. */
      DP_UTIL_API PropertyListChain( const PropertyListChain &rhs );

      /*! \brief Desctructor of a PropertyListChain */
      DP_UTIL_API virtual ~PropertyListChain();

      /** \brief Add a new PropertyList to the chain. All properties of the given propertyList will be visible at the end of the current list,
                 but not duplicated.
          \param propertyList The PropertyList to add. */
      DP_UTIL_API void addPropertyList(PropertyList *propertyList);

      /*! \brief Get the beginning iterator of PropertyLists.
       *  \return A const beginning iterator of PropertyLists. */
      DP_UTIL_API PropertyListConstIterator beginPropertyLists() const;

      /*! \brief Get the ending iterator of PropertyLists.
       *  \return A const ending iterator of PropertyLists. */
      DP_UTIL_API PropertyListConstIterator endPropertyLists() const;

      /** \brief Remove a PropertyList from the chain
          \param propertyList The PropertyList to remove.
      **/
      DP_UTIL_API void removePropertyList(PropertyList *propertyList);

      /*! \brief Get the number of properties.
       *  \return The total number of properties in this PropertyListChain. */
      DP_UTIL_API unsigned int getPropertyCount() const;

      /*! \brief Get a property by index.
       *  \param index The index of the property to get.
       *  \returns The PropertyId of the property specified by \a index.
       *  \remark If \a index is larger than the number of properties in this PropertyListChain, the returned
       *  PropertyId is zero, that is, invalid. */
      DP_UTIL_API PropertyId getProperty(unsigned int index) const;
      DP_UTIL_API PropertyId getProperty(const std::string &name) const;
      //FIXME delete this line DP_UTIL_API void addProperty(const std::string &name, PropertyId property);
      DP_UTIL_API std::string getPropertyName(unsigned int index) const;
      DP_UTIL_API std::string getPropertyName(const PropertyId propertyId) const;
      DP_UTIL_API virtual bool hasProperty(const PropertyId propertyId) const;
      DP_UTIL_API virtual bool hasProperty(const std::string &name) const;

      DP_UTIL_API virtual PropertyList *clone() const;
      DP_UTIL_API virtual bool isStatic() const;
    protected:
      bool m_isStatic;          //!< Flag specifying if this PropertyListChain is static

      std::vector<PropertyList*> m_propertyLists;   //!< The container of PropertyLists
    };

    /** \brief This class stores reflection information for one object type. There will be exactly one instance of this object per object type.
    **/
    class ReflectionInfo : public PropertyListImpl
    {
    public:
      // TODO decide if required, ensures that app does not crash if not initialized by classes. might get initialized by initStaticProperties.
      ReflectionInfo() : m_className("Unknown") {}
      virtual ~ReflectionInfo() {}

      /*! \brief Get the class name of this ReflectionInfo
       *  \returns The class name of this ReflectionInfo. */
      const char *getClassName() const { return m_className; }

    protected:
      const char *m_className;      //!< The class name of this ReflectionInfo

    private:
      template <typename ObjectType>
      friend class ReflectionStorage;

      friend class Reflection;
    };

    /** \brief This is the base class for all objects which support the Reflection/Property mechanism. It's basically a PropertyListChain which will get 
    Base class for all objects with properties. Use this as base class for all objects which provide properties **/
    class Reflection : public dp::util::Subject
    {
    public:
      class PropertyEvent;

      /*! \brief Default constructor of a Reflection. */
      Reflection()
        : m_propertyLists( nullptr )
        {
        }

      /*! \brief Copy constructor of a Reflection.
       *  \param rhs The Reflection to copy from. */
      DP_UTIL_API Reflection( const Reflection &rhs );

      /*! \brief Destructor of a Reflection. */
      DP_UTIL_API virtual ~Reflection();

      /*! \brief Get the class name of this Reflection.
       *  \returns The class name of this Reflection. */
      DP_UTIL_API const char *getClassName() const;

      /*! \brief Set the value of a property.
       *  \param propertyId The id of the property to set.
       *  \param value The value to set. */
      template <typename T>
      void setValue(PropertyId propertyId, const T &value)
      {
        TypedProperty<T> *property = (TypedProperty<T>*)(propertyId);
        property->setValue(this, value);
      }

      /*! \brief Get the type of a property.
       *  \param propertyId The id of the property.
       *  \returns The Property::Type of the property specified by \a propertyId. */
      Property::Type getPropertyType(PropertyId propertyId) const
      {
        return propertyId->getType(); //(this->*propertyId).getType();
      }

      /*! \brief Get the value of a property.
       *  \param propertyId The id of the property to get.
       *  \returns the value of the property specified by \a propertyId. */
      template <typename T>
      T getValue(PropertyId propertyId) const
      {
        DP_ASSERT( getPropertyType(propertyId) == static_cast<Property::Type>(TypedPropertyEnum<T>::type) );

        const TypedProperty<T> *property = (TypedProperty<T>*)(propertyId);
        T returnValue;
        property->getValue(this, returnValue);
        return returnValue;
      }

      /*! \brief Get the number of properties in this Reflection.
       *  \returns The number of properties in this Reflection. */
      DP_UTIL_API unsigned int getPropertyCount() const;

      /*! \brief Get a property by index.
       *  \param index The index of the property to get.
       *  \returns The PropertyId of the property specified by \a index.
       *  \remark If this Reflection holds less properties than \a index, a zero PropertyId is returned. */
      DP_UTIL_API PropertyId getProperty(unsigned int index) const;

      /*! \brief Get a property by name.
       *  \param name The name of the property to get.
       *  \returns The PropertyId of the property specified by \a name.
       *  \remark If this Reflection does not hold a property name \a name, a zero PropertyId is returned. */
      DP_UTIL_API PropertyId getProperty(const std::string &name) const;

      /** \brief Add a new Property to the PropertyList.
          \param name The name of the property
          \param property The PropertyId of the property (which is a pointer to the property)
      **/
      DP_UTIL_API void addProperty(const std::string &name, PropertyId property);

      /*! \brief Get the name of a property by index
       *  \param index The index of the property.
       *  \returns The name of the property specified by \a index.
       *  \remark If this Reflection holds less than \a index properties , an empty string is returned. */
      DP_UTIL_API std::string getPropertyName(unsigned int index) const;

      /*! \brief Get the name of a property by PropertyId.
       *  \param propertyId The id of the property.
       *  \returns The name of the property specifed by \a propertyId. */
      DP_UTIL_API std::string getPropertyName(const PropertyId propertyId) const;

      /*! \brief Check if a property is part of this Reflection.
       *  \param propertyId The id of the property to check.
       *  \return \c true if the property specified by \a propertyId is part of this Reflection, otherwise \c false. */
      DP_UTIL_API virtual bool hasProperty(const PropertyId propertyId) const;

      /*! \brief Check if a property is part of this Reflection.
       *  \param name The name of the property to check.
       *  \returns \c true if the property named \a name is part of this Reflection, otherwise \c false. */
      DP_UTIL_API virtual bool hasProperty(const std::string &name) const;

      inline Reflection& operator=( const Reflection &rhs )
      {
        dp::util::Subject::operator=( rhs );
        return( *this );
      }

      static void initStaticProperties(dp::util::ReflectionInfo &properties) {};
      static void initReflectionInfo() {}
    protected:
      PropertyListChain* m_propertyLists; //!< List of PropertyLists for this Object

      /** Create read/write property **/
      template <typename ValueType, typename GetType, typename SetType, typename ObjectType, 
                typename TypeTraits<ValueType, GetType>::parameter_type(ObjectType::*get)() const, 
                void(ObjectType::*set)(typename TypeTraits<ValueType, SetType>::parameter_type),
                typename dp::util::Semantic semantic>
      static Property *makeProperty()
      {
        return new TypedPropertyImpl<ValueType, ObjectType, 
                                     FunctorGet<ValueType, typename TypeTraits<ValueType, GetType>::parameter_type, ObjectType, get>,
                                     FunctorSet<ValueType, typename TypeTraits<ValueType, SetType>::parameter_type, ObjectType, set>,
                                     semantic, true>;
      }

      /** Create read/write member property **/
      template <typename ValueType, typename ObjectType, ValueType ObjectType::*Member, bool readonly, typename dp::util::Semantic semantic>
      static Property *makePropertyMember()
      {
        return new TypedPropertyImpl<ValueType, ObjectType, 
                                     FunctorGetMember<ValueType, ObjectType, Member>,
                                     FunctorSetMember<ValueType, ObjectType, Member, readonly>,
                                     semantic, true>;
      }

      /** Create read/write enum property **/
      template <typename ValueType, typename GetType, typename SetType, typename ObjectType, 
                typename TypeTraits<ValueType, GetType>::parameter_type(ObjectType::*get)() const, 
                void(ObjectType::*set)(typename TypeTraits<ValueType, SetType>::parameter_type),
                typename dp::util::Semantic semantic>
      static Property *makeEnumProperty()
      {
        return new TypedPropertyImplEnum<ValueType, ObjectType, 
                                     FunctorGetEnum<ValueType, typename TypeTraits<ValueType, GetType>::parameter_type, ObjectType, get>,
                                     FunctorSetEnum<ValueType, typename TypeTraits<ValueType, SetType>::parameter_type, ObjectType, set>,
                                     semantic, true>;
      }


      /** Create readonly property **/
      template <typename ValueType, typename GetType, typename ObjectType, 
                typename TypeTraits<ValueType, GetType>::parameter_type(ObjectType::*get)() const, 
                typename dp::util::Semantic semantic>
      static Property *makeProperty()
      {
        return new TypedPropertyImpl<ValueType, ObjectType, 
                                     FunctorGet<ValueType, typename TypeTraits<ValueType, GetType>::parameter_type, ObjectType, get>,
                                     FunctorSetInvalid<ValueType, ObjectType>,
                                     semantic, true>;
      }


      /*! \brief Initialize reflection of a class.
       *  \param classname The name of the class to initialize. */
    public:
      template <typename ObjectType>
      static dp::util::ReflectionInfo* initReflection(const char *classname)
      {
        dp::util::ReflectionInfo *info = ObjectType::getInternalReflectionInfo();
        info->m_className = classname;
        return info;
      }

      DP_UTIL_API static ReflectionInfo* getInternalReflectionInfo();
    protected:
      DP_UTIL_API virtual ReflectionInfo* getReflectionInfo() const;
    private:

      template <typename ObjectType>
      friend class InitReflection;
    };

    class Reflection::PropertyEvent : public dp::util::Event
    {
    public:
      PropertyEvent( Reflection const* source , dp::util::PropertyId propertyId )
        : Event( dp::util::Event::PROPERTY )
        , m_source( source )
        , m_propertyId( propertyId )
      {
      }

      Reflection const* getSource() const { return m_source; }
      dp::util::PropertyId getPropertyId() const { return m_propertyId; }
    private:
      Reflection const*    m_source;
      dp::util::PropertyId m_propertyId;
    };

    /*! \brief Initialize Reflection framework. */
    DP_UTIL_API void initReflection();

    /*! \brief Shut down Reflection framework. */
    DP_UTIL_API void shutdownReflection();

    /** Create a static property for the given PropertType and register it in the ReflectionInfo storage **/
    template <typename ObjectType, typename PropertyType>
    dp::util::Property *createStaticProperty( const char *name )
    {
      Property *property = new PropertyType();
      ObjectType::getInternalReflectionInfo()->addProperty( name, property );
      return property;
    }

    template <typename ObjectType, typename ObjectTypeBase>
    bool deriveStaticProperties()
    {
      ReflectionInfo *info = ObjectType::getInternalReflectionInfo();
      ReflectionInfo *infoBase = ObjectTypeBase::getInternalReflectionInfo();
      unsigned int count = infoBase->getPropertyCount();
      for ( unsigned int index = 0; index < count; ++index )
      {
        PropertyId propertyId = infoBase->getProperty( index );
        propertyId->addRef();
        info->addProperty( infoBase->getPropertyName(propertyId), propertyId );
      }
      return true;
    }

  } // namespace util
} // namespace dp

/** Macros to make life a little bit easier **/

//when adding properties the variable properties must be a reference to a PropertyList and Klass a typedef of the class of the function members
#define REFLECTION_INFO( MyKlass ) \
  static dp::util::ReflectionInfo* m_reflectionInfo;\
  static void initReflectionInfo(); \
  static dp::util::ReflectionInfo* getInternalReflectionInfo(); \
  virtual dp::util::ReflectionInfo* getReflectionInfo() const { return MyKlass::getInternalReflectionInfo(); }

#define REFLECTION_INFO_API( REFLECTION_API, MyKlass ) \
  static REFLECTION_API dp::util::ReflectionInfo* m_reflectionInfo;\
  static REFLECTION_API void initReflectionInfo(); \
  static REFLECTION_API dp::util::ReflectionInfo* getInternalReflectionInfo(); \
  virtual dp::util::ReflectionInfo* getReflectionInfo() const { return getInternalReflectionInfo(); }

#define REFLECTION_INFO_TEMPLATE_API( REFLECTION_API, MyKlass ) \
  static REFLECTION_API dp::util::ReflectionInfo* m_reflectionInfo;\
  static REFLECTION_API void initReflectionInfo(); \
  static REFLECTION_API dp::util::ReflectionInfo* getInternalReflectionInfo(); \
  virtual dp::util::ReflectionInfo* getReflectionInfo() const { return MyKlass::getInternalReflectionInfo(); }

#define INIT_REFLECTION_INFO( Klass ) \
  dp::util::ReflectionInfo * Klass::m_reflectionInfo = dp::util::Reflection::initReflection< Klass >(#Klass); \
  dp::util::ReflectionInfo* Klass::getInternalReflectionInfo() { \
    static dp::util::ReflectionInfo *reflectionInfo = 0; \
    if (!m_reflectionInfo) { \
      m_reflectionInfo = new dp::util::ReflectionInfo(); \
      initReflectionInfo( ); \
    }; \
    return reflectionInfo; \
  } \
  void Klass::initReflectionInfo() { \
  }

#define BEGIN_REFLECTION_INFO( Klass ) \
  dp::util::ReflectionInfo * Klass::m_reflectionInfo = dp::util::Reflection::initReflection< Klass >(#Klass); \
  dp::util::ReflectionInfo* Klass::getInternalReflectionInfo() { \
    if (!m_reflectionInfo) { \
      m_reflectionInfo = new dp::util::ReflectionInfo(); \
      initReflectionInfo( ); \
    }; \
    return m_reflectionInfo; \
  } \
  void Klass::initReflectionInfo() {

#define END_REFLECTION_INFO }

#define BEGIN_REFLECTION_INFO_TEMPLATE( Klass ) \
  template<> void Klass::initReflectionInfo() {

#define END_REFLECTION_INFO_TEMPLATE( Klass ) \
  } \
  template<> dp::util::ReflectionInfo* Klass::m_reflectionInfo = dp::util::Reflection::initReflection< Klass >(#Klass); \
  template<> dp::util::ReflectionInfo* Klass::getInternalReflectionInfo() { \
  static dp::util::ReflectionInfo *reflectionInfo = 0; \
    if (!m_reflectionInfo) { \
      m_reflectionInfo = new dp::util::ReflectionInfo(); \
      initReflectionInfo( ); \
    }; \
    return m_reflectionInfo; \
  }

/*
 * Use ADD_PROPERTY_RW for get/set properties
 * Use ADD_PROPERTY_RW_BOOL for is/set properties
 * Use ADD_PROPERTY_RO for get properties
 */

#define BEGIN_REFLECT_STATIC_PROPERTIES( Klass ) \
  template<> dp::util::ReflectionInfo* Klass::m_reflectionInfo = dp::util::Reflection::initReflection< Klass >(#Klass); \
  template<> dp::util::ReflectionInfo* Klass::getInternalReflectionInfo() { \
    static dp::util::ReflectionInfo *reflectionInfo = 0; \
    if (!reflectionInfo) { \
      reflectionInfo = new dp::util::ReflectionInfo(); \
      initStaticProperties( *reflectionInfo ); \
    }; \
    return reflectionInfo; \
  } \
  static void Klass::initStaticProperties(dp::util::ReflectionInfo &properties) { \
    typedef MyKlass Klass;

#define ADD_PROPERTY_RW(Name, Type, Semantic, GetType, SetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, dp::util::SetType, Klass, &Klass::get##Name, &Klass::set##Name, dp::util::Semantic>())
#define ADD_PROPERTY_RW_BOOL(Name, Type, Semantic, GetType, SetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, dp::util::SetType, Klass, &Klass::is##Name, &Klass::set##Name, dp::util::Semantic>())
#define ADD_PROPERTY_RO(Name, Type, Semantic, GetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, Klass, &Klass::get##Name, dp::util::Semantic>())
#define ADD_PROPERTY_RO_BOOL(Name, Type, Semantic, GetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, Klass, &Klass::is##Name, dp::util::Semantic>())
#define ADD_PROPERTY_RW_ENUM(Name, Type, Semantic, GetType, SetType) properties.addProperty(#Name, makeEnumProperty<Type, dp::util::GetType, dp::util::SetType, Klass, &Klass::get##Name, &Klass::set##Name, dp::util::Semantic>())
#define ADD_PROPERTY_RW_MEMBER(Name, Type, Semantic) properties.addProperty(#Name, makePropertyMember<Type, Klass, &Klass::m_prop##Name, true, dp::util::Semantic>())
#define ADD_PROPERTY_RO_MEMBER(Name, Type, Semantic) properties.addProperty(#Name, makePropertyMember<Type, Klass, &Klass::m_prop##Name, false, dp::util::Semantic>())

#define ADD_STATIC_PROPERTY_RW(Name, Type, Semantic, GetType, SetType) Property *property = makeProperty<Type, dp::util::GetType, dp::util::SetType, Klass, &Klass::get##Name, &Klass::set##Name, dp::util::Semantic>(); properties.addProperty(#Name, property); PID_##Name = property;
#define ADD_STATIC_PROPERTY_RW_BOOL(Name, Type, Semantic, GetType, SetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, dp::util::SetType, Klass, &Klass::is##Name, &Klass::set##Name, dp::util::Semantic>())
#define ADD_STATIC_PROPERTY_RO(Name, Type, Semantic, GetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, Klass, &Klass::get##Name, dp::util::Semantic>())
#define ADD_STATIC_PROPERTY_RO_BOOL(Name, Type, Semantic, GetType) properties.addProperty(#Name, makeProperty<Type, dp::util::GetType, Klass, &Klass::is##Name, dp::util::Semantic>())
#define ADD_STATIC_PROPERTY_RW_ENUM(Name, Type, Semantic, GetType, SetType) properties.addProperty(#Name, makeEnumProperty<Type, dp::util::GetType, dp::util::SetType, Klass, &Klass::get##Name, &Klass::set##Name, dp::util::Semantic>())
#define ADD_STATIC_PROPERTY_RW_MEMBER(Name, Type, Semantic) properties.addProperty(#Name, makePropertyMember<Type, Klass, &Klass::m_prop##Name, true, dp::util::Semantic>())
#define ADD_STATIC_PROPERTY_RO_MEMBER(Name, Type, Semantic) properties.addProperty(#Name, makePropertyMember<Type, Klass, &Klass::m_prop##Name, false, dp::util::Semantic>())

#define END_REFLECT_STATIC_PROPERTIES() }

#define BEGIN_DECLARE_STATIC_PROPERTIES static void initStaticProperties(dp::util::ReflectionInfo &properties) {}
#define END_DECLARE_STATIC_PROPERTIES 

#define DECLARE_STATIC_PROPERTY(Name) static dp::util::PropertyId PID_##Name
#define DEFINE_STATIC_PROPERTY( Klass, Name ) dp::util::PropertyId Klass::PID_##Name = nullptr;

/** RW **/
#define INIT_STATIC_PROPERTY_RW(Klass, Name, Type, Semantic, GetType, SetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::get##Name>, \
  dp::util::FunctorSet<Type, dp::util::TypeTraits<Type, dp::util::SetType>::parameter_type, Klass, &Klass::set##Name>, \
  dp::util::Semantic, true> \
  >(#Name)

#define INIT_STATIC_PROPERTY_RW_TEMPLATE(Klass, Name, Type, Semantic, GetType, SetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::get##Name>, \
  dp::util::FunctorSet<Type, dp::util::TypeTraits<Type, dp::util::SetType>::parameter_type, Klass, &Klass::set##Name>, \
  dp::util::Semantic, true> \
  >(#Name)

/** RW BOOL **/
#define INIT_STATIC_PROPERTY_RW_BOOL(Klass, Name, Type, Semantic, GetType, SetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::is##Name>, \
  dp::util::FunctorSet<Type, dp::util::TypeTraits<Type, dp::util::SetType>::parameter_type, Klass, &Klass::set##Name>, \
  dp::util::Semantic, true> \
  >(#Name)

#define INIT_STATIC_PROPERTY_RW_BOOL_TEMPLATE(Klass, Name, Type, Semantic, GetType, SetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::is##Name>, \
  dp::util::FunctorSet<Type, dp::util::TypeTraits<Type, dp::util::SetType>::parameter_type, Klass, &Klass::set##Name>, \
  dp::util::Semantic, true> \
  >(#Name)

/** RW ENUM **/
#define INIT_STATIC_PROPERTY_RW_ENUM(Klass, Name, Type, Semantic, GetType, SetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImplEnum<Type, Klass, \
  dp::util::FunctorGetEnum<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::get##Name>, \
  dp::util::FunctorSetEnum<Type, dp::util::TypeTraits<Type, dp::util::SetType>::parameter_type, Klass, &Klass::set##Name>, \
  dp::util::Semantic, true> \
  >(#Name)

#define INIT_STATIC_PROPERTY_RW_ENUM_TEMPLATE(Klass, Name, Type, Semantic, GetType, SetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImplEnum<Type, Klass, \
  dp::util::FunctorGetEnum<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::get##Name>, \
  dp::util::FunctorSetEnum<Type, dp::util::TypeTraits<Type, dp::util::SetType>::parameter_type, Klass, &Klass::set##Name>, \
  dp::util::Semantic, true> \
  >(#Name)

/** RO **/
#define INIT_STATIC_PROPERTY_RO(Klass, Name, Type, Semantic, GetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::get##Name>, \
  dp::util::FunctorSetInvalid<Type, Klass>, \
  dp::util::Semantic, true> \
  >(#Name)

#define INIT_STATIC_PROPERTY_RO_TEMPLATE(Klass, Name, Type, Semantic, GetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::get##Name>, \
  dp::util::FunctorSetInvalid<Type, Klass>, \
  dp::util::Semantic, true> \
  >(#Name)

/** RO BOOL **/
#define INIT_STATIC_PROPERTY_RO_BOOL(Klass, Name, Type, Semantic, GetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::is##Name>, \
  dp::util::FunctorSetInvalid<Type, Klass>, \
  dp::util::Semantic, true> \
  >(#Name)

#define INIT_STATIC_PROPERTY_RO_BOOL_TEMPLATE(Klass, Name, Type, Semantic, GetType)  \
  Klass::PID_##Name = \
  dp::util::createStaticProperty<Klass, \
  dp::util::TypedPropertyImpl<Type, Klass, \
  dp::util::FunctorGet<Type, dp::util::TypeTraits<Type, dp::util::GetType>::parameter_type, Klass, &Klass::is##Name>, \
  dp::util::FunctorSetInvalid<Type, Klass>, \
  dp::util::Semantic, true> \
  >(#Name)

#define INIT_STATIC_PROPERTY_RW_MEMBER(Klass, Name, Type, Semantic) \
  Klass::PID_##Name = makePropertyMember<Type, Klass, &Klass::m_prop##Name, true, dp::util::Semantic>(); \
  Klass::getInternalReflectionInfo()->addProperty( #Name, Klass::PID_##Name )

#define INIT_STATIC_PROPERTY_RO_MEMBER(Klass, Name, Type, Semantic) \
  Klass::PID_##Name = makePropertyMember<Type, Klass, &Klass::m_prop##Name, false, dp::util::Semantic>(); \
  Klass::getInternalReflectionInfo()->addProperty( #Name, Klass::PID_##Name )




#define DUMMY(X)

// HELPERS
#define DERIVE_TOKENPASTE( x, y ) x ## y
#define DERIVE_TOKENPASTE2( x, y ) DERIVE_TOKENPASTE( x, y)

// NOTE, edit and continue must not be enabled since it converts __LINE__ to a function. __COUNTER__ can be used instead once we support gcc 4.3.
#define DERIVE_STATIC_PROPERTIES(Klass, Base ) static bool DERIVE_TOKENPASTE2(deriveStaticProperties, __LINE__ )  = dp::util::deriveStaticProperties<Klass, Base>();
#define DERIVE_STATIC_PROPERTIES_TEMPLATE(Klass, Base ) static bool DERIVE_TOKENPASTE2(deriveStaticProperties, __LINE__ )  = dp::util::deriveStaticProperties<Klass, Base>();

/* Example of a reflected class. Note that dp::sg::core::Object is already derived from Reflection.
   Objects derived from dp::sg::core::Object or subclasses of dp::sg::core::Object must not derive from
   Reflection

class MyReflectedClass : public Reflection
{
public:
  MyReflectedClass()
  {
    // this must be called in each constructor of the class
    // note that this is a noop upon the second call
    INIT_REFLECTION( MyReflectedClass );
  }
  // usual class stuff
  
  void setValue( int value );
  int getValue() const;

  void setBoolValue( bool value );
  bool isBoolValue() const;

  void setVec3fValue( const Vec3f &value );
  const Vec3f &getVec3fValue() const; // return by reference

  void setVec3fValue2( const Vec3f &value );
  Vec3f getVec3fValue2() const; // return by value!

  Vec2f getMaxSize() const;
  bool isSupported() const;

  // here are some examples how to initialize those static properties
  BEGIN_REFLECT_STATIC_PROPERTIES( Material )
  //          property name,        datatype, semantic information, set parameter type, get return type
  ADD_PROPERTY_RW     (Value,       int,      SEMANTIC_VALUE,       value,              value           );
  ADD_PROPERTY_RW_BOOL(BoolValue,   bool,     SEMANTIC_VALUE,       value,              value           );
  ADD_PROPERTY_RW     (Vec3fValue,  Vec3f,    SEMANTIC_VALUE,       const_reference,    const_reference);
  ADD_PROPERTY_RW     (Vec3fValue2, Vec3f,    SEMANTIC_VALUE,       const_reference,    value);
  ADD_PROPERTY_RO     (MaxSize,     Vec2f,    SEMANTIC_VALUE,       value);
  ADD_PROPERTY_RO_BOOL(Supported,   bool,     SEMANTIC_VALUE,       value);
  
  END_REFLECT_STATIC_PROPERTIES()
};

*/
