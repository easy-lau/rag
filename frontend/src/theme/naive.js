/*
 * Shared Naive UI theme overrides.
 *
 * Naive UI derives several peer colors in JavaScript. Passing `var(--token)`
 * into `common` would therefore make its color parser fail. Read the resolved
 * CSS token values when the current color mode changes instead; the CSS file
 * remains the single palette source for normal DOM and teleported overlays.
 */
function readToken(name) {
  if (typeof window === 'undefined' || !document.documentElement) {
    return `var(--ui-${name})`
  }

  return getComputedStyle(document.documentElement).getPropertyValue(`--ui-${name}`).trim()
}

export function createNaiveThemeOverrides() {
  const token = readToken
  const border = (name = 'border') => `1px solid ${token(name)}`
  const popoverTheme = {
    borderRadius: token('radius-popover'),
    color: token('surface-raised'),
    dividerColor: token('divider'),
    textColor: token('text'),
    boxShadow: token('shadow-float')
  }

  return {
    common: {
      primaryColor: token('primary'),
      primaryColorHover: token('primary-hover'),
      primaryColorPressed: token('primary-pressed'),
      primaryColorSuppl: token('primary'),
      infoColor: token('info'),
      infoColorHover: token('info-hover'),
      infoColorPressed: token('info-pressed'),
      infoColorSuppl: token('info'),
      successColor: token('success'),
      successColorHover: token('success-hover'),
      successColorPressed: token('success-pressed'),
      successColorSuppl: token('success'),
      warningColor: token('warning'),
      warningColorHover: token('warning-hover'),
      warningColorPressed: token('warning-pressed'),
      warningColorSuppl: token('warning'),
      errorColor: token('danger'),
      errorColorHover: token('danger-hover'),
      errorColorPressed: token('danger-pressed'),
      errorColorSuppl: token('danger'),
      textColorBase: token('text'),
      textColor1: token('text'),
      textColor2: token('text-secondary'),
      textColor3: token('text-tertiary'),
      textColorDisabled: token('text-disabled'),
      placeholderColor: token('placeholder'),
      placeholderColorDisabled: token('text-disabled'),
      iconColor: token('icon'),
      iconColorHover: token('text-secondary'),
      iconColorPressed: token('text'),
      iconColorDisabled: token('text-disabled'),
      borderColor: token('border'),
      dividerColor: token('divider'),
      bodyColor: token('bg'),
      cardColor: token('surface'),
      modalColor: token('surface-raised'),
      popoverColor: token('surface-raised'),
      inputColor: token('surface'),
      inputColorDisabled: token('surface-disabled'),
      tableColor: token('surface'),
      tableHeaderColor: token('surface-muted'),
      actionColor: token('surface-muted'),
      hoverColor: token('surface-hover'),
      pressedColor: token('surface-pressed'),
      buttonColor2: token('surface-muted'),
      buttonColor2Hover: token('surface-hover'),
      buttonColor2Pressed: token('surface-pressed'),
      boxShadow1: token('shadow-card'),
      boxShadow2: token('shadow-float'),
      boxShadow3: token('shadow-dialog'),
      fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif',
      fontWeight: '500',
      fontWeightStrong: '600',
      borderRadius: token('radius-control'),
      borderRadiusSmall: '8px'
    },

    Button: {
      heightTiny: token('control-height-tiny'),
      heightSmall: token('control-height-compact'),
      heightMedium: token('control-height'),
      heightLarge: token('control-height-large'),
      borderRadiusTiny: '8px',
      borderRadiusSmall: token('radius-control'),
      borderRadiusMedium: token('radius-control'),
      borderRadiusLarge: token('radius-control'),
      fontWeight: '500',
      fontWeightStrong: '600'
    },

    Input: {
      heightTiny: token('control-height-tiny'),
      heightSmall: token('control-height-compact'),
      heightMedium: token('control-height'),
      heightLarge: token('control-height-large'),
      borderRadius: token('radius-control'),
      color: token('surface'),
      colorFocus: token('surface'),
      colorDisabled: token('surface-disabled'),
      textColor: token('text'),
      textColorDisabled: token('text-disabled'),
      placeholderColor: token('placeholder'),
      placeholderColorDisabled: token('text-disabled'),
      caretColor: token('primary'),
      border: border(),
      borderHover: border('border-strong'),
      borderFocus: border('border-focus'),
      borderDisabled: border(),
      boxShadowFocus: token('focus-ring')
    },

    Select: {
      menuBoxShadow: token('shadow-float'),
      peers: {
        InternalSelection: {
          heightTiny: token('control-height-tiny'),
          heightSmall: token('control-height-compact'),
          heightMedium: token('control-height'),
          heightLarge: token('control-height-large'),
          borderRadius: token('radius-control'),
          color: token('surface'),
          colorActive: token('surface'),
          colorDisabled: token('surface-disabled'),
          textColor: token('text'),
          textColorDisabled: token('text-disabled'),
          placeholderColor: token('placeholder'),
          placeholderColorDisabled: token('text-disabled'),
          caretColor: token('primary'),
          arrowColor: token('icon'),
          border: border(),
          borderHover: border('border-strong'),
          borderActive: border('border-focus'),
          borderFocus: border('border-focus'),
          boxShadowActive: token('focus-ring'),
          boxShadowFocus: token('focus-ring'),
          peers: { Popover: popoverTheme }
        },
        InternalSelectMenu: {
          borderRadius: token('radius-popover'),
          color: token('surface-raised'),
          groupHeaderTextColor: token('text-tertiary'),
          optionTextColor: token('text'),
          optionTextColorActive: token('primary'),
          optionTextColorDisabled: token('text-disabled'),
          optionCheckColor: token('primary'),
          optionColorPending: token('surface-hover'),
          optionColorActive: token('primary-subtle'),
          actionDividerColor: token('divider')
        }
      }
    },

    Popover: {
      ...popoverTheme,
      padding: '10px 12px'
    },

    Dropdown: {
      borderRadius: token('radius-popover'),
      color: token('surface-raised'),
      dividerColor: token('divider'),
      optionTextColor: token('text'),
      optionTextColorHover: token('text'),
      optionTextColorActive: token('primary'),
      optionTextColorChildActive: token('primary'),
      optionColorHover: token('surface-hover'),
      optionColorActive: token('primary-subtle'),
      padding: '4px',
      peers: { Popover: popoverTheme }
    },

    Popconfirm: {
      peers: { Popover: popoverTheme }
    },

    Card: {
      color: token('surface'),
      colorModal: token('surface-raised'),
      colorPopover: token('surface-raised'),
      colorTarget: token('surface'),
      colorEmbedded: token('surface'),
      colorEmbeddedModal: token('surface-raised'),
      colorEmbeddedPopover: token('surface-raised'),
      textColor: token('text'),
      titleTextColor: token('text'),
      borderColor: token('border'),
      actionColor: token('surface-muted'),
      titleFontWeight: '600',
      boxShadow: token('shadow-card'),
      borderRadius: token('radius-card')
    },

    Modal: {
      color: token('surface-raised'),
      textColor: token('text'),
      boxShadow: token('shadow-dialog')
    },

    Dialog: {
      color: token('surface-raised'),
      textColor: token('text'),
      titleTextColor: token('text'),
      border: border(),
      borderRadius: token('radius-dialog'),
      titleFontWeight: '600'
    },

    Drawer: {
      color: token('surface-raised'),
      textColor: token('text'),
      titleTextColor: token('text'),
      borderRadius: token('radius-dialog'),
      headerBorderBottom: border('divider'),
      footerBorderTop: border('divider'),
      titleFontWeight: '600',
      boxShadow: token('shadow-dialog')
    },

    Message: {
      color: token('surface-raised'),
      colorInfo: token('surface-raised'),
      colorSuccess: token('surface-raised'),
      colorError: token('surface-raised'),
      colorWarning: token('surface-raised'),
      colorLoading: token('surface-raised'),
      textColor: token('text'),
      textColorInfo: token('text'),
      textColorSuccess: token('text'),
      textColorError: token('text'),
      textColorWarning: token('text'),
      textColorLoading: token('text'),
      border: border(),
      borderRadius: token('radius-popover'),
      boxShadow: token('shadow-float')
    },

    Notification: {
      color: token('surface-raised'),
      textColor: token('text'),
      headerTextColor: token('text'),
      descriptionTextColor: token('text-secondary'),
      borderRadius: token('radius-card'),
      headerFontWeight: '600',
      boxShadow: token('shadow-float')
    },

    Form: {
      labelTextColor: token('text-secondary'),
      labelFontWeight: '600'
    },

    Tag: {
      borderRadius: token('radius-pill'),
      fontWeightStrong: '600'
    },

    Scrollbar: {
      color: token('border-strong'),
      colorHover: token('text-tertiary'),
      railColor: 'transparent',
      borderRadius: token('radius-pill')
    }
  }
}

export default createNaiveThemeOverrides
