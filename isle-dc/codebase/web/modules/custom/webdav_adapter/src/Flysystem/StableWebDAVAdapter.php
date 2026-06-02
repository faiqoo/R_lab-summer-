<?php

namespace Drupal\webdav_adapter\Flysystem;

use League\Flysystem\WebDAV\WebDAVAdapter;
use League\Flysystem\Adapter\Polyfill\NotSupportingVisibilityTrait;
use League\Flysystem\Adapter\Polyfill\StreamedCopyTrait;

class StableWebDAVAdapter extends WebDAVAdapter {

    use StreamedCopyTrait;
    use NotSupportingVisibilityTrait;

}
