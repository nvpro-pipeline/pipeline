
void main(void)
{
  // in the transparent pass we just need to call the (special) emitColor, in opaque pass we need to check alpha and possibly discard
  if ( ! sys_TransparentPass )
  {
    float alpha;

    if ( gl_FrontFacing )
    {
      alpha = frontOpacity;
    }
    else
    {
      alpha = backOpacity;
    }

    if ( textureEnable )
    {
      vec4 texColor = texture2D( sampler, varTexCoord0 );
      switch ( envMode )
      {
        case TEM_REPLACE:
          alpha = texColor.a;
          break;
        default:      
        case TEM_MODULATE:
          alpha *= texColor.a;
          break;
      }
    }

    switch ( alphaFunction )
    {
      case AF_NEVER:                                     discard;            // Never draw the fragment
      case AF_LESS:     if ( alphaThreshold <= alpha ) { discard; } break;   // Draw the fragment if fragment.a <  threshold
      case AF_EQUAL:    if ( alphaThreshold != alpha ) { discard; } break;   // Draw the fragment if fragment.a == threshold
      case AF_LEQUAL:   if ( alphaThreshold <  alpha ) { discard; } break;   // Draw the fragment if fragment.a <= threshold
      case AF_GREATER:  if ( alpha <= alphaThreshold ) { discard; } break;   // Draw the fragment if fragment.a >  threshold
      case AF_NOTEQUAL: if ( alpha == alphaThreshold ) { discard; } break;   // Draw the fragment if fragment.a != threshold
      case AF_GEQUAL:   if ( alpha <  alphaThreshold ) { discard; } break;   // Draw the fragment if fragment.a >= threshold
      default:
      case AF_ALWAYS:                                               break;   // Always draw the fragment
    }

    if ( !lightingEnabled )
    {
      alpha = unlitColor.a;
    }

    // discard any transparent stuff
    if ( alpha < 1.0f )
    {
      discard;
    }
  }

  emitColor( vec4(1) );
}
