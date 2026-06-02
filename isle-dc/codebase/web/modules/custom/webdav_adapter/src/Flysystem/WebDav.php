<?php

namespace Drupal\webdav_adapter\Flysystem;
use League\Flysystem\Filesystem;
use League\Flysystem\WebDAV\WebDAVAdapter;
//use OrangeJuice\Flysystem\WebDAV\WebDAVAdapter;
use Sabre\DAV\Client as WebDavClient; // TODO COmment back in if line below is shit
use Drupal\webdav_adapter\Flysystem\StableWebDAVAdapter; // TODO Remove if is hit

use Drupal\Core\Language\LanguageManagerInterface;
use Drupal\Core\Logger\LoggerChannelInterface;
use Drupal\Core\Logger\RfcLogLevel;
use Drupal\Core\Plugin\ContainerFactoryPluginInterface;
use Drupal\Core\Url;
use Drupal\flysystem\Plugin\FlysystemPluginInterface;
use Drupal\flysystem\Plugin\FlysystemUrlTrait;
use Drupal\jwt\Authentication\Provider\JwtAuth;
use GuzzleHttp\Exception\ConnectException;
use GuzzleHttp\HandlerStack;
use Psr\Http\Message\RequestInterface;
use Symfony\Component\DependencyInjection\ContainerInterface;
use Symfony\Component\Mime\MimeTypeGuesserInterface;
use Symfony\Component\HttpFoundation\RequestStack;
// Custom import
use Drupal\flysystem\Annotation\Adapter;

/**
 * Drupal plugin for the WebDAV Flysystem adapter.
 *
 * @Adapter(id = "webdav")
 */
class WebDav implements FlysystemPluginInterface, ContainerFactoryPluginInterface {

  use FlysystemUrlTrait;

  /**
   * The current request.
   *
   * @var \Symfony\Component\HttpFoundation\Request
   */
  protected $request;

  /**
   * Logger.
   *
   * @var \Drupal\Core\Logger\LoggerChannelInterface
   */
  protected $logger;

  protected array $configuration;

  protected WebDavClient $client;
  /**
   * Constructs a WebDAV plugin for Flysystem.
   *
   * @param \Drupal\Core\Logger\LoggerChannelInterface $logger
   *   The fedora adapter logger channel.
   */
  public function __construct(
    WebDavClient $client,
    array $configuration,
    LoggerChannelInterface $logger,
  ) {
    $this->client = $client;
    $this->configuration = $configuration;
    $this->logger = $logger;
  }

  /**
   * {@inheritdoc}
   */
  public static function create(ContainerInterface $container, array $configuration, $plugin_id, $plugin_definition) {
    $client = new WebDavClient([
      'baseUri' => $configuration['base_uri'],
      'userName' => $configuration['username'],
      'password' => $configuration['password'],
      'authType' => 1,
    ]);

  return new static(
    $client,
    $configuration,
    $container->get('logger.channel.flysystem'),
  );
  }

  /**
   * {@inheritdoc}
   */
  public function getAdapter() {
    $root = $this->configuration['root'] ?? '';
    return new StableWebDAVAdapter($this->client, $root); // , FALSE); //, $this->configuration['root'] ?? '/');
  }

  /**
   * {@inheritdoc}
   */
  public function ensure($force = FALSE) {
    // Check fedora root for sanity.
    try {
      /**
      $this->client->propFind(
        $this->configuration['root'] ?? '/',
        [],
        0
      );
      */
      $this->client->options($this->configuration['base_uri'] ?? '/');
    }
    catch (\Throwable $e) {
      return [[
        'severity' => RfcLogLevel::ERROR,
        'message' => 'WebDAV endpoint unavailable: @message',
        'context' => ['@message' => $e->getMessage()],
      ]];
    }
    return [];
  }

}
