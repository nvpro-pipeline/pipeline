float hue( in vec3 rgb )
{
  float max = mdl_math_maxValue( rgb );
  float min = mdl_math_minValue( rgb );
  float range = max - min;
  float inv_range = 1.0 / range;

  float hue = ( range != 0.0 ) ? ( 1.0 / 6.0) * ( ( max == rgb.x ) ? ( ( rgb.y - rgb.z ) * inv_range )
                                                : ( max == rgb.y ) ? ( 2.0 + ( rgb.z - rgb.x ) * inv_range )
                                                                   : ( 4.0 + ( rgb.x - rgb.y ) * inv_range ) )
                               : 0.0;
  return( ( 0.0 <= hue ) ? hue : ( hue + 1.0 ) );
}

float saturation( in vec3 rgb )
{
  float max = mdl_math_maxValue( rgb );
  return( ( max != 0.0 ) ? ( 1.0 - mdl_math_minValue( rgb ) / max ) : 0.0 );
}

vec3 HSVToRGB( in float h, in float s, in float v )
{
  // A hue of 1.0 is questionably valid, and need to be interpreted as of 0.0
  float hPrime = ( h != 1.0 ) ? h * 6.0 : 0.0;
  float hFloor = floor( hPrime );
  float f = hPrime - hFloor;
  float zy = v * s;
  float a = v - zy;
  float b = v - zy * f;
  float c = a + zy * f;

  switch( int( hFloor ) )
  {
    default:
        // debug::assert(!"hue out of [0,1] range");
        // fall through...
    case 0:
      return( vec3( v, c, a ) );
    case 1:
      return( vec3( b, v, a ) );
    case 2:
      return( vec3( a, v, c ) );
    case 3:
      return( vec3( a, b, v ) );
    case 4:
      return( vec3( c, a, v ) );
    case 5:
      return( vec3( v, a, b ) );
    }
}

vec3 blendColor( in vec3 base, in vec3 layer, in float weight, in int mode )
{
  vec3 result = layer;
  switch( mode )
  {
    case color_layer_add :
      result += base;
      break;
    case color_layer_multiply :
      result *= base;
      break;
    case color_layer_screen :
      result += base - base * layer;
      break;
    case color_layer_overlay :
      {
        vec3 mul = base * layer;
        vec3 add = base + layer;
        if ( 0.5 <= base.x )  mul.x = add.x - mul.x - 0.5;
        if ( 0.5 <= base.y )  mul.y = add.y - mul.y - 0.5;
        if ( 0.5 <= base.z )  mul.z = add.z - mul.z - 0.5;
        result = 2 * mul;
      }
      break;
    case color_layer_brightness :
      {
        float baseBrightness  = mdl_math_maxValue( base );
        float layerBrightness = mdl_math_maxValue( layer );
        result = ( baseBrightness == 0 ) ? vec3( layerBrightness ) : base * ( layerBrightness / baseBrightness );
      }
      break;
    case color_layer_color:
      {
        float baseBrightness = mdl_math_maxValue( base );
        float layerBrightness = mdl_math_maxValue( layer );
        result = ( layerBrightness == 0 ) ? vec3( baseBrightness ) : layer * ( baseBrightness / layerBrightness );
      }
      break;
    case color_layer_exclusion :
      result = base + layer - 2 * base * layer;
      break;
    case color_layer_average :
      result = 0.5 * ( base + layer );
      break;
    case color_layer_lighten :
      result = max( base, layer );
      break;
    case color_layer_darken :
      result = min( base, layer );
      break;
    case color_layer_sub :
      result = base + layer - 1;
      break;
    case color_layer_negation :
      result = 1 - abs( 1 - ( base + layer ) );
      break;
    case color_layer_difference :
      result = abs( layer - base );
      break;
    case color_layer_softlight :
      result = vec3( ( layer.x < 0.5 ) ? 2.0 * ( layer.x * base.x + base.x * base.x * ( 0.5 - layer.x ) ) : 2.0 * ( sqrt( base.x ) * ( layer.x - 0.5 ) + base.x - layer.x * base.x )
                   , ( layer.y < 0.5 ) ? 2.0 * ( layer.y * base.y + base.y * base.y * ( 0.5 - layer.y ) ) : 2.0 * ( sqrt( base.y ) * ( layer.y - 0.5 ) + base.y - layer.y * base.y )
                   , ( layer.z < 0.5 ) ? 2.0 * ( layer.z * base.z + base.z * base.z * ( 0.5 - layer.z ) ) : 2.0 * ( sqrt( base.z ) * ( layer.z - 0.5 ) + base.z - layer.z * base.z ) );
      break;
    case color_layer_colordodge :
      result = vec3( ( layer.x == 1.0 ) ? 1.0 : min( base.x / ( 1.0 - layer.x ), 1.0 )
                   , ( layer.y == 1.0 ) ? 1.0 : min( base.y / ( 1.0 - layer.y ), 1.0 )
                   , ( layer.z == 1.0 ) ? 1.0 : min( base.z / ( 1.0 - layer.z ), 1.0 ) );
      break;
    case color_layer_reflect :
      result = vec3( ( layer.x == 1.0 ) ? 1.0 : min( base.x * base.x / ( 1.0 - layer.x ), 1.0 )
                   , ( layer.y == 1.0 ) ? 1.0 : min( base.y * base.y / ( 1.0 - layer.y ), 1.0 )
                   , ( layer.z == 1.0 ) ? 1.0 : min( base.z * base.z / ( 1.0 - layer.z ), 1.0 ) );
      break;
    case color_layer_colorburn :
      result = vec3( ( layer.x == 0.0 ) ? 0.0 : max( 1.0 - ( 1.0 - base.x ) / layer.x, 0.0 )
                   , ( layer.y == 0.0 ) ? 0.0 : max( 1.0 - ( 1.0 - base.y ) / layer.y, 0.0 )
                   , ( layer.z == 0.0 ) ? 0.0 : max( 1.0 - ( 1.0 - base.z ) / layer.z, 0.0 ) );
      break;
    case color_layer_phoenix :
      result = min( base, layer ) - max( base, layer ) + 1.0;
      break;
    case color_layer_hardlight :
      {
        vec3 mul = base * layer;
        vec3 add = base + layer;
        if ( 0.5 < layer.x )  mul.x = add.x - mul.x - 0.5;
        if ( 0.5 < layer.y )  mul.y = add.y - mul.y - 0.5;
        if ( 0.5 < layer.z )  mul.z = add.z - mul.z - 0.5;
        result = 2 * mul;
      }
      break;
    case color_layer_pinlight :
      result = vec3( ( ( ( 0.5 < layer.x ) && ( base.x < layer.x ) ) || ! ( ( 0.5 < layer.x ) || ( base.x < layer.x ) ) ) ? layer.x : base.x
                   , ( ( ( 0.5 < layer.y ) && ( base.y < layer.y ) ) || ! ( ( 0.5 < layer.y ) || ( base.y < layer.y ) ) ) ? layer.y : base.y
                   , ( ( ( 0.5 < layer.z ) && ( base.z < layer.z ) ) || ! ( ( 0.5 < layer.z ) || ( base.z < layer.z ) ) ) ? layer.z : base.z );
      break;
    case color_layer_hardmix :
      result = vec3( ( base.x + layer.x <= 1.0 ) ? 0.0 : 1.0
                   , ( base.y + layer.y <= 1.0 ) ? 0.0 : 1.0
                   , ( base.z + layer.z <= 1.0 ) ? 0.0 : 1.0 );
      break;
    case color_layer_lineardodge :
      result = min( base + layer, 1.0 );
      break;
    case color_layer_linearburn :
      result = max( base + layer - 1.0, 1.0 );
      break;
    case color_layer_spotlight :
      result = 2 * base * layer;
      break;
    case color_layer_spotlightblend :
      result = base * layer + base;
      break;
    case color_layer_hue :
      result = HSVToRGB( hue( layer ), saturation( base ), mdl_math_maxValue( base ) );
      break;
    case color_layer_saturation :
      result = HSVToRGB( hue( base ), saturation( layer ), mdl_math_maxValue( base ) );
      break;
    default :
      break;
  }
  return( mix( base, result, weight ) );
}

float blendMono( in int monoSource, in vec3 color )
{
  switch( monoSource )
  {
    case mono_average :
      return( average( color ) );
    case mono_luminance :
      return( mdl_math_luminance( color ) );
    case mono_maximum :
    default :
      return( mdl_math_maxValue( color ) );
  }
}

_base_textureReturn mdl_base_blendColorLayers( in vec3 base, in int monoSource )
{
  _base_textureReturn tr;
  tr.tint = base;
  tr.mono = blendMono( monoSource, tr.tint );
  return( tr );
}

_base_textureReturn mdl_base_blendColorLayers( in _base_colorLayer layer, in vec3 base, in int monoSource )
{
  _base_textureReturn tr;
  tr.tint = blendColor( base, layer.layerColor, layer.weight, layer.mode );
  tr.mono = blendMono( monoSource, tr.tint );
  return( tr );
}

_base_textureReturn mdl_base_blendColorLayers( in _base_colorLayer[1] layers, in vec3 base, in int monoSource )
{
  _base_textureReturn tr;
  tr.tint = blendColor( base, layers[0].layerColor, layers[0].weight, layers[0].mode );
  tr.mono = blendMono( monoSource, tr.tint );
  return( tr );
}

_base_textureReturn mdl_base_blendColorLayers( in _base_colorLayer[2] layers, in vec3 base, in int monoSource )
{
  _base_textureReturn tr;
  tr.tint = base;
  for ( int i=0 ; i<2 ; i++ )
  {
    tr.tint = blendColor( tr.tint, layers[i].layerColor, layers[i].weight, layers[i].mode );
  }
  tr.mono = blendMono( monoSource, tr.tint );
  return( tr );
}

