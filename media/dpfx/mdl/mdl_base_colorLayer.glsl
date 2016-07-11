_base_colorLayer mdl_base_colorLayer()
{
  _base_colorLayer cl;
  cl.layerColor = vec3( 0.0f, 0.0f, 0.0f );
  cl.weight     = 1.0f;
  cl.mode       = color_layer_blend;
  return( cl );
}

_base_colorLayer mdl_base_colorLayer( in vec3 layerColor, in float weight, in int mode )
{
  _base_colorLayer cl;
  cl.layerColor = layerColor;
  cl.weight     = weight;
  cl.mode       = mode;
  return( cl );
}

